#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import itertools
import missingno as msno
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')  


# In[2]:


superstore=pd.read_csv('C:/Users/HP/Downloads/supermarket3.csv')


# In[3]:


superstore.head(5)


# #### CREATE NEW COLUMN PROFIT

# In[4]:


superstore['Profit']=superstore.apply(lambda row : row.Total-row.cogs,axis=1)


# In[5]:


superstore.head()


# In[6]:


superstore.rename(columns={'Invoice ID':'Invoiceid','Customer type':'Customertype','Product line':'Productline','Unit price':'Unitprice','Tax 5%':'Tax5%','gross margin percentage':'Grossmargin%,','gross income':'Grossincome','cogs':'Cogs','Order Date':'Orderdate'},inplace = True)


# In[7]:


superstore.head(5)


# In[8]:


superstore.dtypes


# In[9]:


superstore['Year']=pd.DatetimeIndex(superstore.Orderdate).year


# In[10]:


superstore.head(5)


# In[11]:


superstore['Month']=pd.DatetimeIndex(superstore.Orderdate).month


# In[12]:


superstore.head()


# In[13]:


superstore['Day']=pd.DatetimeIndex(superstore.Orderdate).day


# In[14]:


superstore.head()


# In[ ]:





# ### lets make the columns consiatent

# In[15]:


superstore.rename(columns={'Invoice ID':'Invoiceid','Customer type':'Customertype','Product line':'Productline','Unit price':'Unitprice','Tax 5%':'Tax5%','gross margin percentage':'Grossmargin%,','gross income':'Grossincome','cogs':'Cogs'},inplace = True)


# In[16]:


superstore.rename(columns={'Total':'Totalsales'},inplace = True)


# In[17]:


superstore.head(2)


# ### LETS STUDY AND UNDERSTAND THE DATA

# In[18]:


superstore.index


# In[19]:


superstore.sample


# In[20]:


superstore.columns


# In[21]:


list(superstore.columns)


# In[22]:


superstore.size


# In[23]:


superstore.shape


# In[24]:


print('number of columns',superstore.shape[1])


# In[25]:


print('number of rows',superstore.shape[0])


# In[26]:


superstore.duplicated().any()


# In[27]:


superstore.info()


# In[28]:


superstore.isnull().sum()


# In[29]:


superstore.dropna(inplace=True)


# In[30]:


superstore.isnull().sum()


# In[31]:


superstore.count()


# In[32]:


superstore.describe()


# In[33]:


superstore.hist(bins=10, figsize=(18,10))


# In[34]:


sns.pairplot(superstore, hue='Totalsales')


# In[35]:


plt.figure(figsize=(20,20))
sns.heatmap(superstore.corr(), vmin=-1, cmap="plasma_r", annot=True)


# In[36]:


superstore.head(2)


# ### Questions to be answered
# 
# ### 1,What day of the week are our sales high on average?
# 
# ### 2What is monthly average sales?
# 
# ### 3,What are the year over growth year/ total sales over the years?
# 
# ### 4,Who are the top customer of all times?
# 
# ### 5,What are the highest sales by category?
# 
# ### 6,Which city does have the highest sales?

# In[37]:


superstore.head(5)


# In[ ]:





# ### Q1,What day of the week are our sales high on average?
# 

# In[38]:


superstore.head(2)


# In[ ]:





# In[39]:


dailysales = superstore.groupby('Day').agg({'Totalsales':"sum"}).reset_index()
#I grouped  my dataset by "Order Date" and I want that according to "Sales" sum.
dailysales


# In[40]:


plt.title('AVERAGE DAILY SALES')
sns.scatterplot(data=dailysales,x='Day', y='Totalsales')
plt.xticks(rotation=45);


# ### Q2, What is monthly average sales?

# In[41]:


monthlysales = superstore.groupby('Month').agg({'Totalsales':"sum"}).reset_index()


# In[42]:


monthlysales


# In[43]:


superstore['Month'].unique()


# In[44]:


monthlysales=monthlysales.groupby('Month').agg({'Totalsales': 'sum'}).reset_index()
ax= sns.barplot(x='Month', y='Totalsales', data=monthlysales, color="crimson").set(title="Total sales over months");


# ### Q3,What are the year over growth year/ total sales over the years?

# In[45]:


superstore['Year'].unique()


# In[46]:


yearlysales = superstore.groupby('Year').agg({'Totalsales':"sum"}).reset_index()


# In[47]:


yearlysales


# In[48]:


yearlysales=yearlysales.groupby('Year').agg({'Totalsales': 'sum'}).reset_index()
ax= sns.barplot(x='Year', y='Totalsales', data=yearlysales,).set(title="Total sales over Year");


# In[49]:


superstore.head(2)


# ### Q4,Who are the top customer of all times?

# In[50]:


super_top_customer = superstore.groupby(['Customertype', 'Year']).agg({'Totalsales':'sum'}).reset_index()

#px.bar(data_top_cust_year[data_top_cust_year['year']== 2015], x='Customer Name', y='Sales',title='top customer 2015')
#Here, I filtered only the information of all sales in 2015.


# In[51]:


super_top_customer


# In[52]:


branch_type = superstore.groupby('Branch').agg({'Totalsales':"sum"}).reset_index()


# In[53]:


branch_type


# In[54]:


ax= sns.barplot(x='Branch', y='Totalsales', data=branch_type,).set(title="Total sales by Branch");


# ### Q5, What are the highest sales by Productline

# In[55]:


product_line = superstore.groupby('Productline').agg({'Totalsales':"sum"}).reset_index()


# In[56]:


product_line


# In[57]:


ax= sns.barplot(x='Productline', y='Totalsales', data=product_line,).set(title="Total sales by Productline")


# In[58]:


product_line = superstore.groupby('Productline').agg({'Profit':"sum"}).reset_index()


# In[59]:


product_line


# In[60]:


ax= sns.barplot(x='Productline', y='Profit', data=product_line,).set(title="Total profit by Productline")


# 

# ### Q6,Which city does have the highest sales

# In[61]:


city_sales = superstore.groupby('City').agg({'Totalsales':"sum"}).reset_index()


# In[62]:


city_sales


# In[63]:


ax= sns.barplot(x='City', y='Totalsales', data=city_sales,).set(title="Total sales by City")


# In[64]:


superstore.head(2)


# In[65]:


sns.countplot(x='Gender', data=superstore, palette='tab10')


# In[66]:


customer_type= superstore.groupby(['Customertype'])[['Totalsales', 'Profit']].mean()


# In[67]:


customer_type


# In[68]:


colors_4 = ['magenta','yellow','orange','red']
colors_3 = ['green', 'blue','cyan']


# In[69]:


customer_type.plot.pie(subplots=True, figsize=(20,10), labels=customer_type.index, autopct='%1.1f%%', colors=colors_3)
plt.show()


# In[ ]:





# In[70]:


branch_type= superstore.groupby(['Branch'])[['Totalsales', 'Profit']].mean()


# In[71]:


branch_type


# In[72]:


branch_type.sort_values('Profit')[['Profit','Totalsales']].plot(kind='bar', figsize=(10,7))


# In[73]:


superstore.head(1)


# In[ ]:





# In[74]:


branch_type.sum(axis=0)


# In[75]:


customer_type.sum()


# In[76]:


superstore.head(2)


# In[77]:


superstore['Branch'].unique()


# In[78]:


superstore['Productline'].unique()


# # TIME SERIES FORCASTING

# In[79]:


superstore.head()


# In[80]:


superstore3=superstore.drop(['Customertype','Invoiceid','Productline','Gender','Unitprice','Quantity','Tax5%','Grossmargin%,','Grossincome','Rating','Date','Payment'],axis=1)


# In[81]:


superstore4=superstore3.drop(['Time','Cogs','Year','Month','Day'],axis=1)


# In[82]:


superstore4.head()


# In[83]:


superstore4['Branchcity']=superstore4['Branch']+'-'+superstore4['City']


# In[84]:


superstore4.head()


# In[86]:


#droping the old columns 
superstore4.drop(['Branch','City'],axis=1,inplace =True)
superstore4.head()


# In[87]:


#Convert Order date column into year month format
superstore4['Orderdate'] = pd.to_datetime(superstore4['Orderdate']).dt.to_period('m')
#data= data.sort_values(by=['Order Date'])
superstore4.head()


# In[89]:


#Change date format into timestamps
superstore4['Orderdate'] = superstore4['Orderdate'].astype(str)
superstore4['Orderdate']=pd.to_datetime(superstore4['Orderdate'])


# In[90]:


superstore4.head(2)


# In[91]:


#group data for using it for forecasting and applying timeseries modelling
superstore4=superstore4[['Orderdate','Totalsales']].groupby('Orderdate').sum()


# In[92]:


superstore4.head()


# In[93]:


#plot time series data
superstore4.plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Retail Giant Sales')
plt.show(block=False)


# ### 2, SPLIT DATA INTO TRAIN AND TEST SET

# In[95]:


train_len=42
train=superstore4[0:train_len]
test=superstore4[train_len:]


# ### 3, TIME SERIES DECOMPOSITION

# In[96]:


#Additive method

from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(superstore4.Totalsales, model='additive') 
fig = decomposition.plot()
plt.show()


# In[97]:


##Multiplicative method
decomposition = sm.tsa.seasonal_decompose(superstore4.Totalsales, model='multiplicative') 
fig = decomposition.plot()
plt.show()


# ### 4. Build and Evaluate Time Series forecasting model

# ### Simple Time series method

# ### 1. Naive method

# In[99]:


y_hat_naive = test.copy()
y_hat_naive['naive_forecast'] = train['Totalsales'][train_len-1]


# In[100]:


#Plot train test and forecast

plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()


# ### Comments:
# - As seen in the plot,the naive method uses the last or previous month data which is 2014-06. We can see that the forecast for the next six months is the same value as the last observation .

# In[102]:


#Calculate RMSE and MAPE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_naive['naive_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_naive['naive_forecast'])/test['Totalsales'])*100,2)

results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 2. Simple Average method

# In[103]:


y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Totalsales'].mean()


# In[104]:


#Plot train test and forecast

plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Simple average forecast')
plt.legend(loc='best')
plt.title('Simple Average Method')
plt.show()


# In[105]:


#Calculate RMSE and MAPE

rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_avg['avg_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_avg['avg_forecast'])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple average method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 3. Simple moving average method

# In[108]:


y_hat_sma = superstore4.copy()
ma_window = 3
y_hat_sma['sma_forecast'] = superstore4['Totalsales'].rolling(ma_window).mean()
y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]


# In[110]:


#Plot train test and forecast

plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()


# In[112]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_sma['sma_forecast'][train_len:])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_sma['sma_forecast'][train_len:])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 5,Simple exponential smoothing

# In[113]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model = SimpleExpSmoothing(train['Totalsales'])
model_fit = model.fit(optimized=True)
model_fit.params
y_hat_ses = test.copy()
y_hat_ses['ses_forecast'] = model_fit.forecast(6)


# In[114]:


#Plot train test and forecast
plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Simple Exponential Smoothing Method')
plt.show()


# In[115]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_ses['ses_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_ses['ses_forecast'])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results


# ### 5. Holt's Exponential Smoothing

# In[117]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train['Totalsales']) ,seasonal_periods=12 ,trend='additive', seasonal=None)
model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)
print(model_fit.params)
y_hat_holt = test.copy()
y_hat_holt['holt_forecast'] = model_fit.forecast(len(test))


# In[118]:


#Plot train test and forecast
plt.figure(figsize=(12,4))
plt.plot( train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Holt\'s Exponential Smoothing Method')
plt.show()


# In[119]:


#Calculate RSME and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_holt['holt_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_holt['holt_forecast'])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 6. Holt Winters' additive method

# In[120]:


y_hat_hwa = test.copy()
model = ExponentialSmoothing(np.asarray(train['Totalsales']) ,seasonal_periods=12 ,trend='add', seasonal='add')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwa['hw_forecast'] = model_fit.forecast(6)


# In[121]:


#Plot train test and forecast
plt.figure(figsize=(12,4))
plt.plot( train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Additive Method')
plt.show()


# In[123]:


#This model captures the level and the trend along with the seasonality very well.
#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_hwa['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_hwa['hw_forecast'])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 7,Holt Winter's multiplicative method

# In[124]:


y_hat_hwm = test.copy()
model = ExponentialSmoothing(np.asarray(train['Totalsales']) ,seasonal_periods=12 ,trend='add', seasonal='mul')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwm['hw_forecast'] = model_fit.forecast(6)


# In[125]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot( train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Mulitplicative Method')
plt.show()


# In[126]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_hwm['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_hwm['hw_forecast'])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 8,Auto Regressive methods

# In[129]:


# Stationarity vs non-stationary time series
superstore4['Totalsales'].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Retail Giant Sales')
plt.show(block=False)


# In[133]:


#Augmented Dickey-Fuller (ADF) test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(superstore4['Totalsales'])

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# In[134]:


# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
from statsmodels.tsa.stattools import kpss
kpss_test = kpss(superstore4['Totalsales'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# In[136]:


#Box Cox transformation to make variance constant
from scipy.stats import boxcox
data_boxcox = pd.Series(boxcox(superstore4['Totalsales'], lmbda=0), index = superstore4.index)

plt.figure(figsize=(12,4))
plt.plot(data_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# In[137]:


#Differencing to remove trend
data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), superstore4.index)
plt.figure(figsize=(12,4))
plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
plt.show()


# In[138]:


data_boxcox_diff.dropna(inplace=True)


# In[139]:


#Augmented Dickey-Fuller (ADF) test
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(data_boxcox_diff)

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values @ 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# In[140]:


# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
from statsmodels.tsa.stattools import kpss
kpss_test = kpss(data_boxcox_diff)

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# In[141]:


train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
test_data_boxcox_diff = data_boxcox_diff[train_len-1:]


# ## 1. Auto regression method (AR)

# In[143]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) 
model_fit = model.fit()
print(model_fit.params)


# In[144]:


#Recover original time series
y_hat_ar = data_boxcox_diff.copy()
y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[0])
y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])


# In[145]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_ar['ar_forecast'][test.index.min():], label='Auto regression forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.show()


# In[146]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_ar['ar_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_ar['ar_forecast'][test.index.min():])/test['Totalsales'])*100,2)


# In[147]:


tempResults = pd.DataFrame({'Method':['Autoregressive (AR) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 2. Moving average method (MA)

# In[148]:


model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) 
model_fit = model.fit()
print(model_fit.params)


# In[149]:


#Recover original time series
y_hat_ma = data_boxcox_diff.copy()
y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum()
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(data_boxcox[0])
y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])


# In[151]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(superstore4['Totalsales'][:train_len], label='Train')
plt.plot(superstore4['Totalsales'][train_len:], label='Test')
plt.plot(y_hat_ma['ma_forecast'][test.index.min():], label='Moving average forecast')
plt.legend(loc='best')
plt.title('Moving Average Method')
plt.show()


# In[153]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_ma['ma_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_ma['ma_forecast'][test.index.min():])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Moving Average (MA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 3. Auto regression moving average method (ARMA)

# In[154]:


model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.params)


# In[155]:


#Recover original time series
y_hat_arma = data_boxcox_diff.copy()
y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(data_boxcox[0])
y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])


# In[157]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot( superstore4['Totalsales'][:train_len-1], label='Train')
plt.plot(superstore4['Totalsales'][train_len-1:], label='Test')
plt.plot(y_hat_arma['arma_forecast'][test.index.min():], label='ARMA forecast')
plt.legend(loc='best')
plt.title('ARMA Method')
plt.show()


# In[158]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_arma['arma_forecast'][train_len-1:])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_arma['arma_forecast'][train_len-1:])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive moving average (ARMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 4. Auto regressive integrated moving average (ARIMA)¶

# In[159]:


model = ARIMA(train_data_boxcox, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.params)


# In[160]:


#Recover original time series forecast
y_hat_arima = data_boxcox_diff.copy()
y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(data_boxcox[0])
y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])


# In[161]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_arima['arima_forecast'][test.index.min():], label='ARIMA forecast')
plt.legend(loc='best')
plt.title('Autoregressive integrated moving average (ARIMA) method')
plt.show()


# In[162]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_arima['arima_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_arima['arima_forecast'][test.index.min():])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ### 5. Seasonal auto regressive integrated moving average (SARIMA)

# In[163]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) 
model_fit = model.fit()
print(model_fit.params)


# In[164]:


#Recover original time series forecast
y_hat_sarima = data_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])


# In[165]:


#Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train['Totalsales'], label='Train')
plt.plot(test['Totalsales'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][test.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')
plt.show()


# In[166]:


#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Totalsales'], y_hat_sarima['sarima_forecast'][test.index.min():])).round(2)
mape = np.round(np.mean(np.abs(test['Totalsales']-y_hat_sarima['sarima_forecast'][test.index.min():])/test['Totalsales'])*100,2)

tempResults = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:




