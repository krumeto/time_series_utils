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


