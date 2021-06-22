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



# lockdown1 = pd.DataFrame({
#   'holiday': 'lockdown1',
#   'ds': pd.to_datetime(["2020-03-11"]),
#   'upper_window': 68,
# })

# m = Prophet(holidays=lockdown1)

def clipping(df, lower, upper):
  df.y = df.y.clip(lower=df.y.quantile(lower), upper=df.y.quantile(upper))
  return df

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


