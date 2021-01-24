import pandas as pd
import numpy as np

class SameForecaster:
    """A class to return the same value for a time series with very few data points.
    Assumes the Prophet format of ds, y for the time series and the y.
    
    """

    def __init__(self):
        self.mean_daily_value = None
        self.history_dates = None
        self.days_to_one = None
        self.is_fitted = False
  
    def fit(self, time_series):
        #TODO - find missing dates in time_series
        self.history_dates = time_series.ds
        self.mean_daily_value = time_series.y.fillna(0).mean()
        self.days_to_one = round(1/self.mean_daily_value)
        self.is_fitted = True

  
    def make_future_dataframe(self, periods, freq='D', include_history = True):
        
        if not is_fitted:
            print('Model is not fitted yet. Please call fit first.')

        last_date = self.history_dates.max()

        dates = pd.date_range(
                start=last_date,
                periods=periods + 1,  # An extra in case we include start
                freq=freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:periods]  # Return correct number of periods

        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))

        return pd.DataFrame({'ds': dates})

    def predict(self, future_df):

        if not is_fitted:
            print('Model is not fitted yet. Please call fit first.')

        future_df_copy = future_df.copy()
        future_df_copy["yhat"] = 0
        future_df_copy["yhat"].iloc[::self.days_to_one] = 1

        return future_df_copy