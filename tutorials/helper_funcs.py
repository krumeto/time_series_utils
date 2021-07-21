import pandas as pd
import matplotlib.pyplot as plt
import glob

def load_energy_data(data_path):
    """
  Takes the path to the Total Load files as an input.
  Returns a concatinated df, containing the start of each interval, actuals and a forecast
  
    """
    #Load and concat the files
    total_load = pd.DataFrame()
    for file in glob.glob(data_path+"*"):
        if 'Total Load' in file:
            partial_df = pd.read_csv(file)
            total_load = pd.concat([total_load, partial_df])

    #Rename the columns
    total_load.columns = ['time_interval', 'forecast', 'actuals']

    #Keep the start of the interval only and change type to datetime
    total_load['time'] = total_load.time_interval.str.split("-", expand=True)[0]

    total_load.time = pd.to_datetime(total_load.time)
    #Change type of actuals to numeric, replacing non-numbers with np.NaN
    total_load.actuals = pd.to_numeric(total_load.actuals, errors='coerce')

    total_load = total_load.drop('time_interval', axis=1)
    total_load['date_object'] = total_load.time.dt.date

    #Keep until incl. 2020
    total_load = total_load.loc[lambda df: df['time'] < "2021-01-01"]

    return total_load[['time', 'actuals', 'forecast', 'date_object']]

def load_temperature(file_path):
    """Read the temperature data and return Sofia averages"""
    temps = pd.read_csv(file_path, usecols=['day', 'София'], parse_dates=['day'])
    temps.columns = ['day', 'avg_temp']
    temps['date_object'] = temps.day.dt.date

    return temps

def merge_datasets(energy_df, temp_df):
    full_df = energy_df.merge(temp_df, on='date_object', how='left')
    
    return full_df.loc[:, ['time', 'actuals', 'avg_temp']]

def resample_as_needed(df, freq = 'D'):
    return df.set_index('time').resample(freq).mean()

def add_first_day_of_month(df, col):
  df[f'{col}_mm'] = df[col] + pd.offsets.Day() - pd.offsets.MonthBegin()
  return df

def get_month_end(date):
  return pd.to_datetime(date) - pd.offsets.Day() + pd.offsets.MonthEnd()

def get_last_n_month_ends(date, n, slide =1):
  return [get_month_end(date) - pd.offsets.MonthEnd(i+slide) for i in range(n)]

def clipping(df, lower, upper):
  df.y = df.y.clip(lower=df.y.quantile(lower), upper=df.y.quantile(upper))
  return df

def get_first_days(df):
    return df.assign(month_start = ((df['ds'].dt.day < 6)).astype(int))

# lockdown1 = pd.DataFrame({
#   'holiday': 'lockdown1',
#   'ds': pd.to_datetime(["2020-03-11"]),
#   'upper_window': 68,
# })

# m = Prophet(holidays=lockdown1)

def clipping(df, lower, upper):
  df.y = df.y.clip(lower=df.y.quantile(lower), upper=df.y.quantile(upper))
  return df

def _create_dataframe(df):
  data_date_range = pd.date_range(start=df["discharge_date"].min(), end = min(df["discharge_date"].max(), pd.to_datetime(date.today())), freq='d')
  
  return pd.DataFrame({'ds':data_date_range})

def get_last_days(df):
    return df.assign(month_end = (df.ds == (df.ds + MonthEnd(0) )).astype(int) )


def filter_data(grouped_df, facility, patient_type,group_col = 'patient_type' ):
  
  sample = grouped_df.loc[
                 (grouped_df["facility"] == facility) & 
                 (grouped_df[group_col] == patient_type)]
  
  y = sample.set_index('discharge_date').accounts
  y.name= 'y'
  
  output = _create_dataframe(sample).merge(y,
                                    left_on = 'ds',
                                    right_index=True,
                                    how='left'
                                  )
  output.y = output.y.fillna(0)
  
  return output


def plot_combo_forecast(grouped_data, combo, start_date):
  
  test_series = filter_data(grouped_data, combo[0], combo[1])

  m = Prophet()
  m.add_country_holidays('US')
  m.add_regressor('month_end')

  m.fit(test_series.loc[test_series.ds > start_date].pipe(get_last_days))

  future = m.make_future_dataframe(periods=31).pipe(get_last_days)
  
  forecast = m.predict(future)
  forecast['facility'] = combo[0]
  forecast['patient_type'] = combo[1]
  
  for y in ['yhat', 'yhat_lower', 'yhat_upper']:
    forecast[y] = forecast[y].clip(lower=0)
  
  m.plot(forecast);
  m.plot_components(forecast);

  cutoffs = pd.to_datetime(['2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01'])
  df_cv = cross_validation(m, cutoffs=cutoffs, horizon='31 days', parallel = 'processes')
  df_p = performance_metrics(df_cv,rolling_window=1)
  print('#########################################################################')
  print('Cross-val metrics:')
  print(df_p)

  
def filter_forecast(forecast,facility, patient_type):
  X_hat = forecast.loc[(forecast.facility == facility) & (forecast.patient_type == patient_type)]
  subset = X_hat.loc[:, ['ds', 'yhat']]
  subset['ds'] = pd.to_datetime(subset['ds'])
  
  return subset


def compare_forecasts(actuals, forecasts,facility, patient_type, resample=False):
  
  X = filter_data(actuals, facility, patient_type).reset_index(drop='False')
  X_hat = filter_forecast(forecasts, facility, patient_type).reset_index(drop='False')
  
  results = X.merge(
                    X_hat,
                    left_on='ds',
                    right_on='ds', how='outer'  
  )
  
  if resample:
    results = results.set_index('ds').resample('W').sum().reset_index()
  
  error = (abs(results.y - results.yhat)/results.y).mean()
  
  results.set_index('ds').loc["2020-05-31":].plot(title=f"{facility} {patient_type} MAPE: {error:.2%}")
  plt.show()
  results.set_index('ds').iloc[-60:].plot(title=f"Recent 2 months {facility} {patient_type} MAPE: {error:.2%}")
  plt.show()
  
  return results

import os
import yaml

class suppress_stdout_stderr(object):
    """
    https://github.com/joblib/joblib/issues/868
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def add_covid_baseline_to_timeseries(df, timeseries_column):
    """Takes a dataframe and a timeseries column of datetime type and adds to it a covid_baseline 
    where covid months are 1 and all the rest are 0
    
    If lockdown is set to True, April-20 is set to 3 in the series, while March and May are set to 2.
    
    """
    
    if not is_datetime64_any_dtype(df[timeseries_column]):
        print("The provided column is not of datetime format. Results might be unpredictable")
    
    covid_series = ((df[timeseries_column] > pd.to_datetime('2020-03-01')) &
                    (df[timeseries_column] < pd.to_datetime('2021-12-01')) 
                   ).astype('int')
    
    covid_series = covid_series.rename('covid_baseline')
    
    return pd.concat([df,covid_series], axis=1)
  
def add_shutdown_baseline_to_timeseries(df, timeseries_column):
    """Takes a dataframe and a timeseries column of datetime type and adds to it a covid_baseline 
    where covid months are 1 and all the rest are 0
    
    If lockdown is set to True, April-20 is set to 3 in the series, while March and May are set to 2.
    
    """
    
    if not is_datetime64_any_dtype(df[timeseries_column]):
        print("The provided column is not of datetime format. Results might be unpredictable")
    
    covid_series = ((df[timeseries_column] > pd.to_datetime('2020-03-15')) &
                    (df[timeseries_column] < pd.to_datetime('2020-06-15')) 
                   ).astype('int')
    
    covid_series = covid_series.rename('shut_down')
    
    return pd.concat([df,covid_series], axis=1)


if __name__ == "__main__":
     
    pass
    # RESAMPLE = True
    # energy = load_energy_data("./data/")
    # temp = load_temperature('./data/avg_temp.csv')
    # final_df = merge_datasets(energy, temp)
    # print(final_df.info())
    # print(final_df.head())

    # if RESAMPLE:
    #     print(final_df.pipe(resample_as_needed))

    # final_df.pipe(resample_as_needed).plot(subplots=True, figsize=(14,10), grid=True)
    # plt.show()


