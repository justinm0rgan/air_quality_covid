# import necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller

# set directory
PROJECT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_DIR, 'images')
os.makedirs(IMAGES_PATH, exist_ok=True)

# function for saving figures
def save_fig(fig_id, tight_layout=True, fig_extension="png", 
              resolution=300, transparent=False):
                
    '''Function that saves visual as png at a specified resolution
    Enter fig_id to specify what you would like your figure to be called.
    Other paremeters have default values and can be altered if needed'''
    
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    print("To", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, transparent=transparent)
    
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
  plt.xticks(positions, labels, rotation=70, fontsize = 12);
  ax.set_ylabel("PM2.5 (ug/m3)", fontsize=14);
  ax.legend(loc='best', fontsize=12);
  ax.set_title(plot_title, fontsize=20);
  
  # save map
  save_fig(plot_title.lower().replace(" ", "_"))
  plt.show();
  
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
    
# tlcc function
def crosscorr(datax, datay, lag=0, wrap=False):
    ''' Lag-N cross correlation. 
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns: crosscorr : float
    '''
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
    
