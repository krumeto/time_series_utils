import pandas as pd
import numpy as np

class SameForecaster:
    """A class to return the same value for a time series with very few data points.
    Assumes the Prophet format of ds, y for the time series and the y.
    
    """

    def __init__(self, strategy = 'cyclic'):
        self.mean_daily_value = None
        self.history_dates = None
        self.days_to_one = None
        self.is_fitted = False
        self.strategy = strategy
        self.historical_period_values = None
        self.periods = None
  
    def fit(self, time_series, ts_start= None, ts_end=None):
        #TODO - find missing dates in time_series
        
        date_range = pd.period_range(start=time_series.ds.min(), end=time_series.ds.max(), freq='D')
        period_df = pd.DataFrame(date_range, columns = ['ds'])
        period_df.ds = period_df.ds.dt.to_timestamp(how='start')

        #create a dataframe with values
        time_series.ds = pd.to_datetime(time_series.ds)
        full_ts = period_df.merge(time_series.set_index('ds'), left_on='ds', right_index=True, how='left').fillna(0)
        full_ts.columns = ['ds', 'y']
        
        #TODO
        self.history_dates = full_ts.ds
        self.mean_daily_value = full_ts.y.fillna(0).mean()
        self.days_to_one = round(1/self.mean_daily_value)
        self.is_fitted = True
        self.historical_period_values = time_series.y.values

  
    def make_future_dataframe(self, periods, freq='D', include_history = True):
        
        if not self.is_fitted:
            print('Model is not fitted yet. Please call fit first.')

        last_date = self.history_dates.max()
        self.periods = periods

        dates = pd.date_range(
                start=last_date,
                periods=periods + 1,  # An extra in case we include start
                freq=freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:periods]  # Return correct number of periods
      

        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
            dates = pd.to_datetime(dates)

        return pd.DataFrame({'ds': dates})

    def predict(self, future_df):

        if not self.is_fitted:
            print('Model is not fitted yet. Please call fit first.')

        if self.strategy == 'cyclic':
          future_df_copy = future_df.copy()
          future_df_copy["yhat"] = 0
          future_df_copy["yhat"].iloc[::self.days_to_one] = 1

        if self.strategy == 'repeat':
          future_df_copy = future_df.copy()
          
          predictions = self.historical_period_values[-self.periods:]
          future_df_copy["yhat"] = np.concatenate((self.historical_period_values, predictions))
          future_df_copy["yhat"] = future_df_copy["yhat"].fillna(0)

        return future_df_copy