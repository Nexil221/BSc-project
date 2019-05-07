import numpy as np
import pandas as pd
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from fastai.structured import *
import matplotlib.pyplot as plt
import os

import glob


path = r'C:\Users\Marcin\PycharmProjects\inz\venv'
all_files = glob.glob(path + "/*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, sep=',', header=1, low_memory=False) #read csv
    li.append(df)

crypto_data = pd.concat(li, axis=0, ignore_index=True)
#del crypto_data['Unix Timestamp']
crypto_data = crypto_data.drop(columns=['Unix Timestamp'])
crypto_data['Date'] = pd.to_datetime(crypto_data['Date'])
crypto_data['Date'] = pd.to_numeric(crypto_data['Date'])

#btc - 30563
#eth - 55991
#ltc - 60059
#zec - 67655

plt.figure(1, figsize=(12, 10))
plt.subplot(221)
plt.plot(crypto_data.iloc[:30563, 0], crypto_data.iloc[:30563, 2]) #Open
plt.plot(crypto_data.iloc[30564:55991, 0], crypto_data.iloc[30564:55991, 2])
plt.plot(crypto_data.iloc[55992:60059, 0], crypto_data.iloc[55992:60059, 2])
plt.plot(crypto_data.iloc[60060:67655, 0], crypto_data.iloc[60060:67655, 2])
plt.ylabel('USD')
plt.xlabel('Date(Unix Timestamp)')
plt.title('Open')

plt.subplot(222)
plt.plot(crypto_data.iloc[:30563, 0], crypto_data.iloc[:30563, 3]) #Open
plt.plot(crypto_data.iloc[30564:55991, 0], crypto_data.iloc[30564:55991, 3])
plt.plot(crypto_data.iloc[55992:60059, 0], crypto_data.iloc[55992:60059, 3])
plt.plot(crypto_data.iloc[60060:67655, 0], crypto_data.iloc[60060:67655, 3])
plt.ylabel('USD')
plt.xlabel('Date(Unix Timestamp)')
plt.title('High')

plt.subplot(223)
plt.plot(crypto_data.iloc[:30563, 0], crypto_data.iloc[:30563, 4]) #Open
plt.plot(crypto_data.iloc[30564:55991, 0], crypto_data.iloc[30564:55991, 4])
plt.plot(crypto_data.iloc[55992:60059, 0], crypto_data.iloc[55992:60059, 4])
plt.plot(crypto_data.iloc[60060:67655, 0], crypto_data.iloc[60060:67655, 4])
plt.ylabel('USD')
plt.xlabel('Date(Unix Timestamp)')
plt.title('Low')

plt.subplot(224)
plt.plot(crypto_data.iloc[:30563, 0], crypto_data.iloc[:30563, 5]) #Open
plt.plot(crypto_data.iloc[30564:55991, 0], crypto_data.iloc[30564:55991, 5])
plt.plot(crypto_data.iloc[55992:60059, 0], crypto_data.iloc[55992:60059, 5])
plt.plot(crypto_data.iloc[60060:67655, 0], crypto_data.iloc[60060:67655, 5])
plt.ylabel('USD')
plt.xlabel('Date(Unix Timestamp)')
plt.title('Close')

#add_datepart(btc_base, 'Date')
#btc_base['Date'] = pd.to_datetime(btc_base['Date'])
#btc_base['Date'] = pd.to_numeric(btc_base['Date']) #change data to number
#['Symbol'] = 1 #change btc symbol to 1

os.makedirs('tmp', exist_ok=True)
crypto_data.to_feather('tmp/BTCUSD_2018.csv')
crypto_data_feather = pd.read_feather('tmp/BTCUSD_2018.csv')

df, y, nas = proc_df(crypto_data_feather, 'Open')

def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(crypto_data_feather, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

#X_train.shape, y_train.shape, X_valid.shape #Jupyter

def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

df_trn, y_trn, nas = proc_df(crypto_data_feather, 'Open', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 200000)
y_train, _ = split_vals(y_trn, 200000)

m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)

#draw_tree(m.estimators_[0], df_trn, precision=3) #draw tree in IPython(Jupyter)




