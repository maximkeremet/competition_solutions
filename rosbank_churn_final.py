
# coding: utf-8
# Rosbank Churn Compretion
# Author: Maxim Keremet
# Resulting in 0.82 AUC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm,tqdm_notebook

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import confusion_matrix

from catboost import CatBoostClassifier
import lightgbm as lgbm

import gc

## Get data and quick processing

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# concat all pieces and leave clasification target
full = pd.concat([train, test])
full = full.drop('target_sum', axis=1)
full.columns = full.columns.str.lower()

# generate time columns
full['transac_month'] = full['period'].str.split('/').str[1]
full['transac_day'] = full['trdatetime'].str[:2]
full['transac_year'] = '20'+full['trdatetime'].str[5:7]
full['transac_year_month'] = '20'+full['trdatetime'].str[5:7]+'-'+full['period'].str.split('/').str[1]
full['transac_time'] = full['trdatetime'].str[-11:]

# mark test sample
full['target_flag'] = full['target_flag'].fillna(-1).astype(int)

# not needed anymore
full.drop(['period','trdatetime'], axis=1,inplace=True)

# construct the datetime column for filtering 
full['datetime'] = full['transac_day']+'-'+full['transac_month']+'-'+full['transac_year']+' '+full['transac_time']
full['datetime'] = pd.to_datetime(full['datetime'], format='%d-%m-%Y %H:%M:%S:%f')

# proper sort and view
full.sort_values(['cl_id', 'datetime'], inplace=True)
#full.set_index('cl_id', inplace=True)


print('Total unique users: %d' %len(full.index.unique()))
print('Unique currency types: %d' %len(full.currency.unique()))
print('Unique product categories: %s' %len(full.trx_category.unique()))
print('Unique seller codes: %s' %len(full.mcc.unique()))
print('Target distribution:') 
print(round(full.target_flag.value_counts(normalize=True)*100,2))


# identify train, test ids
train_ids = train['cl_id'].unique()
test_ids = test['cl_id'].unique()


gc.collect()


## Making feature counters
transactions= full.copy()
print("---Feature engeneering started.---")

# Transactions counts
last_month_tranaction_count = pd.DataFrame(full.groupby(['cl_id', 'transac_month']).size().unstack().ffill(axis=1).iloc[:, -1])
last_month_tranaction_count.rename(columns={'12':'last_month_transactions'}, inplace=True)
transactions = transactions.merge(last_month_tranaction_count, how='left', left_on='cl_id', right_index=True)
transactions['last_month_transactions'] = transactions['last_month_transactions'].astype(dtype='int64')

transaction_count = dict(full.groupby('cl_id').size())
transactions['transaction_count'] = transactions['cl_id'].map(transaction_count)

# Currency stats
currency_stats = full.groupby(['cl_id', 'currency'])['amount'].agg([np.mean, np.std, np.var]).rename(columns={'mean': 'curr_mean','std': 'curr_std','var': 'curr_var'}).reset_index()
transactions = transactions.merge(currency_stats, how='left', on=['cl_id', 'currency'])
# Currency proportions
currency_proportions = (transactions.groupby(['cl_id', 'currency'])['amount'].sum().unstack().T/transactions.groupby(['cl_id', 'currency'])['amount'].sum().unstack().sum(axis=1)).T

new_columns = []
for i in currency_proportions.columns:
    i = ('curr_%s'%i)
    new_columns.append(i)
    
currency_proportions.columns = new_columns
transactions = transactions.merge(currency_proportions, how='left', left_on='cl_id', right_index=True)

print("---Currency stats constructed.---")

# Cat stats
category_stats = full.groupby(['cl_id', 'trx_category'])['amount'].agg([np.mean, np.std, np.var]).rename(columns={'mean': 'cat_mean','std': 'cat_std','var': 'cat_var'}).reset_index()
transactions = transactions.merge(category_stats, how='left', on=['cl_id', 'trx_category'])
# Cat proportions
category_proportions = (transactions.groupby(['cl_id', 'trx_category'])['amount'].sum().unstack().T/transactions.groupby(['cl_id', 'trx_category'])['amount'].sum().unstack().sum(axis=1)).T

new_columns = []
for i in category_proportions.columns:
    i = ('cat_%s'%i)
    new_columns.append(i)
    
category_proportions.columns = new_columns
transactions = transactions.merge(category_proportions, how='left', left_on='cl_id', right_index=True)

print("---Category stats constructed.---")

# MCC stats
currency_stats = full.groupby(['cl_id', 'mcc'])['amount'].agg([np.mean, np.std, np.var]).rename(columns={'mean': 'mcc_mean','std': 'mcc_std','var': 'mcc_var'}).reset_index()
transactions = transactions.merge(currency_stats, how='left', on=['cl_id', 'mcc'])
print("---MCC stats constructed.---")

# Months used
months_used = pd.get_dummies(full.groupby(['cl_id','transac_month']).size().unstack().count(axis=1), 
               drop_first=False, prefix='months_used')
transactions = transactions.merge(months_used, how='left', left_on='cl_id', right_index=True)

month_amount_stats = full.groupby(['cl_id', 'transac_year_month'])['amount'].agg([np.mean, np.std, np.var]).rename(columns={'mean': 'month_mean','std': 'month_std','var': 'month_var'}).reset_index()
months_series = pd.DataFrame(full.groupby(['cl_id', 'transac_year_month'])['amount'].mean().unstack().count(axis=1)).reset_index()
month_amount_stats = month_amount_stats.merge(months_series, on='cl_id', how='left')
month_amount_stats.rename(columns= {'amount':'mean_month_amount', 0:'months_series'}, inplace=True)
transactions = transactions.merge(month_amount_stats, how='left', on=['cl_id', 'transac_year_month'])
print("---Time stats constructed.---")

# Customer life in days from first given transaction to the last
client_life = dict((full.groupby('cl_id')['datetime'].last() - full.groupby('cl_id')['datetime'].first()).dt.days)
transactions['client_life'] = transactions['cl_id'].map(client_life)
print("---Customer life stats constructed.---")

# Last buy
total_buys = full.groupby(['cl_id','transac_year_month']).size().unstack().sum(axis=1)
buy_ratio = full.groupby(['cl_id','transac_year_month']).size().unstack().ffill(axis=1).iloc[:, -1]/total_buys
transactions['last_buy_ratio'] = transactions['cl_id'].map(buy_ratio)

# Buying frequency
frequency = []
for i in tqdm_notebook(transactions['cl_id'].unique()):
    freq = transactions[transactions['cl_id'] == i]['datetime'].diff().mean().seconds/3600
    frequency.append(freq)
    
mean_frequency = {key:value for key, value in zip(transactions['cl_id'].unique(), frequency)}
transactions['mean_frequency'] = transactions['cl_id'].map(mean_frequency)
print("---Buying stats constructed---")

transactions.drop(['mcc', 'currency','trx_category', 'transac_month', 'transac_day',
       'transac_time', 'transac_year', 'transac_year_month', 'datetime', 'channel_type'], axis=1, inplace=True)

print("---Unnecessary columns dropped.---")


print('Rows total:',transactions.shape[0])
print('Features total:',len(transactions.columns))


## Preproc for algorythm

y = transactions[transactions['cl_id'].isin(train_ids)]['target_flag'].values
X = transactions[transactions['cl_id'].isin(train_ids)].fillna(-999).drop(['cl_id','target_flag'] , axis=1).values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_score = transactions[transactions['cl_id'].isin(test_ids)].fillna(-999).drop(['cl_id','target_flag'] , axis=1).values


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


## Catboost

models = []

np.random.seed(0)
for i in tqdm_notebook(range(50)):

    model = CatBoostClassifier(
        iterations=300,
        depth=10,
        thread_count=12,
        border_count=128,
        learning_rate=0.015,
        random_seed=np.random.randint(10**10),
        eval_metric='AUC',
        verbose=False
    )
    
    model.fit(X_train, y_train)
    models.append(model.copy())


## LGBM

models2 = []

np.random.seed(0)
for i in tqdm_notebook(range(100)):

    model = lgbm.LGBMClassifier(
        random_state=np.random.randint(10**10),
        n_estimators=4600,
        max_depth=15 + np.random.randint(0,10),
        num_leaves=15 + np.random.randint(0,10),
        subsample=0.99868,
        colsample_bytree=0.8022,
        reg_alpha=26.4310,
        reg_lambda=19.7836,
        max_bin=8850,
        objective='binary',
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    models2.append(model)


# ## Combining predictions

predictions = []
for _model in tqdm_notebook(models):
    predictions.append(_model.predict_proba(X_score)[:,1])
    
predictions = np.vstack(predictions).T


predictions2 = []
for _model in tqdm_notebook(models2):
    predictions2.append(_model.predict_proba(X_score)[:,1])
    
predictions2 = np.vstack(predictions2).T

prediction = np.column_stack([predictions, predictions2]).mean(axis=1)


submit = pd.DataFrame({'_ID_':transactions[transactions['cl_id'].isin(test_ids)]['cl_id'],'_VAL_':prediction}, 
                      columns = ['_ID_','_VAL_']) 
submit.groupby('_ID_').mean().reset_index().to_csv('predictions/blending_catboost_lgbm_all_features_28052018.csv', index=False)
#submit.to_csv('predictions/stacking_submission_catboost_lgbm_more_depth.csv', index=False)

