import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key="ds479Ypsx3fJNLk9ivc2"
df = quandl.get("NSE/RCOM")
#print (df.head())
df = df[['Open',  'High',  'Low',  'Close']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change']]
forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)
print(len(df))
forecast_out = int(math.ceil(0.001 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
#print (df.head())
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)

