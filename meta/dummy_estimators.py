import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot
import sys
import traceback
from tutorials.helper_funcs import load_energy_data, load_temperature, merge_datasets, resample_as_needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


class SimpleProphet(Prophet):
  
  """
  A simple class including Covid modelling and US Holidays.
  In addition, predictions are clipped at zero to not allow for negative values.
  
  """
  
  def __init__(self, add_lockdown = True, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    
    if add_lockdown:
      lockdown1 = pd.DataFrame({
        'holiday': 'lockdown1',
        'ds': pd.to_datetime(["2020-03-11"]),
        'upper_window': 111,
        'lower_window':0
      })
      
      lockdown2 = pd.DataFrame({
        'holiday': 'lockdown2',
        'ds': pd.to_datetime(["2020-07-01"]),
        'upper_window': 91,
        'lower_window':0
      })
      
      lockdown3 = pd.DataFrame({
        'holiday': 'lockdown3',
        'ds': pd.to_datetime(["2020-10-01"]),
        'upper_window': 135,
        'lower_window':0
      })
      
      lockdowns = pd.concat([lockdown1, lockdown2, lockdown3])
      self.holidays = lockdowns
      
    self.add_country_holidays(country_name='US')
      
  def predict(self, *args, **kwargs):
    forecast = super(SimpleProphet, self).predict(*args, **kwargs)
    
    for y in ['yhat', 'yhat_lower', 'yhat_upper']:
      forecast[y] = forecast[y].clip(lower=0)
      
    return forecast
  
  
class ProphetStepWise(Prophet):
  
    def __init__(self, add_lockdown = True, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
    
      if add_lockdown:
        lockdown1 = pd.DataFrame({
          'holiday': 'lockdown1',
          'ds': pd.to_datetime(["2020-03-11"]),
          'upper_window': 111,
          'lower_window':0
        })
        
        lockdown2 = pd.DataFrame({
          'holiday': 'lockdown2',
          'ds': pd.to_datetime(["2020-07-01"]),
          'upper_window': 91,
          'lower_window':0
        })
        
        lockdown3 = pd.DataFrame({
          'holiday': 'lockdown3',
          'ds': pd.to_datetime(["2020-10-01"]),
          'upper_window': 135,
          'lower_window':0
        })
      
        lockdowns = pd.concat([lockdown1, lockdown2, lockdown3])
        self.holidays = lockdowns
      
      self.add_country_holidays(country_name='US')
    
    def fit(self, df, **kwargs):
        m = super().fit(df, **kwargs)
        if self.growth == 'flat' and self.changepoints is not None:
            kinit = self.stepwise_growth_init(self.history, self.changepoints)
            self.params['m_'] = kinit[1]
        return self
        

    @staticmethod
    def stepwise_growth_init(df, changepoints):
        k = 0
        if len(changepoints) == 0:
            k = 0
            m = df['y_scaled'].mean()
        else:
            m = []
            last_cp = pd.Timestamp.min
            for cp in changepoints:
                df_  = df[(df.ds > last_cp) & (df.ds <= cp)]['y_scaled']
                m.append(df_.mean())
                last_cp = cp
            cp = changepoints.iloc[-1]
            df_  = df[df.ds > cp]['y_scaled']
            m.append(df_.mean())
        return k, m
    
    @staticmethod
    def stepwise_trend(t, m, changepoint_ts):
        if changepoint_ts[0] == 0:
            m_t = m * np.ones_like(t)
        else:
            m_t = np.ones_like(t)
            last_t_s = changepoint_ts[0]-1
            for s, t_s in enumerate(changepoint_ts):
                indx = (t <= t_s) & (t > last_t_s)
                m_t[indx] = m[s]
                last_t_s = t_s

            t_s = changepoint_ts[-1]
            indx = (t > t_s) 
            m_t[indx] = m[-1]
        return m_t
    
    def sample_predictive_trend(self, df, iteration):
        """Simulate the trend using the extrapolated generative model.
        Parameters
        ----------
        df: Prediction dataframe.
        iteration: Int sampling iteration to use parameters from.
        Returns
        -------
        np.array of simulated trend over df['t'].
        """
        k = self.params['k'][iteration]
        m = self.params['m'][iteration]
        deltas = self.params['delta'][iteration]

        t = np.array(df['t'])
        T = t.max()

        # New changepoints from a Poisson process with rate S on [1, T]
        if T > 1:
            S = len(self.changepoints_t)
            n_changes = np.random.poisson(S * (T - 1))
        else:
            n_changes = 0
        if n_changes > 0:
            changepoint_ts_new = 1 + np.random.rand(n_changes) * (T - 1)
            changepoint_ts_new.sort()
        else:
            changepoint_ts_new = []

        # Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
        lambda_ = np.mean(np.abs(deltas)) + 1e-8

        # Sample deltas
        deltas_new = np.random.laplace(0, lambda_, n_changes)

        # Prepend the times and deltas from the history
        changepoint_ts = np.concatenate((self.changepoints_t,
                                         changepoint_ts_new))
        deltas = np.concatenate((deltas, deltas_new))

        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, changepoint_ts)
        elif self.growth == 'logistic':
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(t, cap, deltas, k, m,
                                            changepoint_ts)
        elif self.growth == 'flat':
            trend = self.stepwise_trend(t, self.params['m_'], self.changepoints_t)

        return trend * self.y_scale + df['floor']
    
    def predict_trend(self, df):
        """Predict trend using the prophet model.
        Parameters
        ----------
        df: Prediction dataframe.
        Returns
        -------
        Vector with trend on prediction dates.
        """
        k = np.nanmean(self.params['k'])
        m = np.nanmean(self.params['m'])
        deltas = np.nanmean(self.params['delta'], axis=0)

        t = np.array(df['t'])
        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, self.changepoints_t)
        elif self.growth == 'logistic':
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(
                t, cap, deltas, k, m, self.changepoints_t)
        elif self.growth == 'flat':
            # constant trend
            trend = self.stepwise_trend(t, self.params['m_'], self.changepoints_t)
        return trend * self.y_scale + df['floor']
      
    def predict(self, *args, **kwargs):
      forecast = super(ProphetStepWise, self).predict(*args, **kwargs)
    
      for y in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[y] = forecast[y].clip(lower=0)
      
      return forecast
