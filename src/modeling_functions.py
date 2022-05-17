# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# dickey fuller function
def dickey_fuller(series):
    dftest = adfuller(series)
    
    # Extract and display test results in a user friendly manner
    dfoutput = pd.Series(dftest[0:4], 
    index=['Test Statistic','p-value','#Lags Used', 
    'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dftest)
    print ('Results of Dickey-Fuller test: ')
    print(dfoutput)
    
# plot function
def plot_time_series(df_col1, series2, label1, label2, plot_title):
  '''Function that takes data from df column
  then takes a series of equal length
  plots two line plots as time series'''

  # set tick format
  years = mdates.YearLocator()   # every year
  months = mdates.MonthLocator()  # every month

  # set labels
  positions = [0,365,730,1095,1460,1826]
  labels = ["2017","2018","2019","2020","2021","2022"]

  # create plot
  fig, ax = plt.subplots();
  ax = sns.lineplot(data= df_col1,
  label=label1,
  color = "#0C6291",
  alpha = 0.8);
  plt.plot(series2, color="#A63446", label=label2);
  ax.set(xticks=[x for x in df.index.values if df.index.get_loc(x)%365==0]);

  #format the ticks
  ax.xaxis.set_major_locator(years);
  ax.xaxis.set_minor_locator(months);
  plt.xticks(positions, labels, rotation=70);
  plt.ylabel("PM2.5 (ug/m3)");
  plt.legend(loc='best');
  plt.title(plot_title);
  plt.show();

# arima evaluation functions
def evaluate_arima_model(series, arima_order):
    '''Evaluate an ARIMA model for a given order (p,d,q) and return RMSE'''
    arima_model = ARIMA(series, order=arima_order)  
    arima_fit = arima_model.fit(disp=-1)  
    rmse = np.sqrt(sum((arima_fit.fittedvalues - series).dropna()**2)/len(series))
    return rmse
  
def evaluate_models(series, p_values, d_values, q_values):
    '''Test different possible combinations of p,d,q values 
    that provide the lowest RMSE 
        on a given series. Returns the best order of pdq'''
    series = series.astype('float32')
    best_score, best_order = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    order = (p, d, q)
                    rmse = evaluate_arima_model(series, order)
                    if rmse < best_score:
                        best_score, best_order = rmse, order
#                     print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA: %s  RMSE=%.3f' % (best_order, best_score))
    return best_order
  
def predict(df, columns):
    '''Takes main dataframe and list of columns, returns dataframe of forecast predictions'''
    pred_df = pd.DataFrame()
    for station in columns:
        best_order = evaluate_models(df[station],range(0, 9),range(0, 1),range(0, 9))
        try:
            arima_model = ARIMA(df[station], order=(best_order)) 
            arima_fit = arima_model.fit()  
            preds = arima_fit.predict(start=df.shape[0],end=df.shape[0]+60)
            pred_df[station] = preds
        except:
            print('FAILED:',station)
    return(pred_df)
