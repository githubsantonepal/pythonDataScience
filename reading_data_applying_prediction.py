import quandl, math
import numpy as np
import pandas as pd
import pickle


#Use sklearn.modelselection instead of sklearn.crossvalidation
#reason is in new version of sklearn crossvalidation is changed to modelselection

from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

#quandl.ApiConfig.api_key = 'm1nUYTYPUCPt-FJBAQvx'
df=quandl.get('CHRIS/MGEX_IH1', authtoken='m1nUYTYPUCPt-FJBAQvx')
#print(df)
print (df.columns.tolist())

df=df[['Open','High', 'Low', 'Last', 'Volume']]
df['HL_PCT']=(df['High']-df['Last'])/df['Last']*100
df['PCT_change']=(df['Last']-df['Open'])/df['Open']*100
df=df[['Last', 'HL_PCT', 'PCT_change', 'Volume']]
print(df.head())
forecast_col='Last'
df.fillna(value=-99999, inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)

x=np.array(df.drop(['label'],1))
x=preprocessing.scale(x)
x_lately=x[-forecast_out:]
x=x[:-forecast_out]

df.dropna(inplace=True)

y=np.array(df['label'])
x_train,x_test, y_train, y_test=model_selection.train_test_split(x,y, test_size=0.2)

# Here you can change the algorithm
clf=LinearRegression(n_jobs=-1)
#clf=svm.SVR(kernel='poly')
clf.fit(x_train,y_train)

#Using pickle to save the training data sets so that we don't have to train it again and again.
#with open('linearregression.pickle','wb') as file:
 #   pickle.dump(clf, file)

#pickle_read=open('linearregression.pickle','rb')
#clf=pickle.load(pickle_read)

accuracy=clf.score(x_test, y_test)
#print(accuracy)
forecast_set=clf.predict(x_lately)
print(forecast_set,accuracy, forecast_out)
df['Forecast']=np.nan

last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


