
# coding: utf-8

# In[153]:

import pandas as pd

data=pd.read_csv("data.csv")
data.head()


# In[154]:

data.describe()


# ### Data Pre-processing

# Converting time features (TS and TO) to seconds

# In[74]:

def get_sec(time_str):
    """Get Seconds from time."""
    try:
        h, m = time_str.split(':')
        return int(h) * 3600 + int(m) * 60
    except AttributeError as e:
        return None
# convert to secs
for col in data.columns[4:]:
    data[col]=data[col].apply(get_sec)


# ### TS (Time of sending email) converted to seconds wrt 00:00 hrs

# In[75]:

slice_TS=data.loc[:, 'TS00':'TO14':2]
slice_TO=data.loc[:, 'TO00':'TO14':2]
slice_TS.head()


# ### TO (Time of opening email) converted to seconds wrt 00:00 hrs

# In[76]:

slice_TO.head()


# In[77]:

slice_TS.columns=range(0,len (slice_TS.columns))
slice_TO.columns=range(0,len (slice_TO.columns))
diff_to_ts=slice_TO-slice_TS


# Get diff between TO and TS matrices

# In[78]:

def update_negative_diff (x):
    if x<0:
        return x+86400
    else:
        return x

# adding 24hrs if diff is negative
for col in diff_to_ts.columns:
    diff_to_ts[col]=diff_to_ts[col].apply(update_negative_diff)


# In[79]:

def get_min(row):
    min_index=0
    for x in range (0,15):
        if row[x]<row[min_index]:
            min_index=x
    return min_index

min_ts_list=[]
for x in range (0,int (diff_to_ts.size/15)):
    min_index=get_min(diff_to_ts.loc[x])
    min_ts_list.append (slice_TS[min_index][x])   


# Selecting minimum of diff as objective is to *minimize(TO-TS)*

# In[80]:

data['min_ts']=min_ts_list


# ### Correlation Matrix

# In[81]:

corr=data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# ### Corr Matrix of processed DF

# In[149]:

corr=data.drop (['TS00','TO00','TS01','TO01','TS02','TO02','TS03','TO03','TS04','TO04','TS05','TO05','TS06','TO06','TS07','TO07','TS08','TO08','TS09','TO09','TS10','TO10','TS11','TO11','TS12','TO12','TS13','TO13','TS14','TO14'], axis=1).corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# ### Feature Selection and Dataset split (Train, Eval)

# In[82]:

from sklearn.model_selection import train_test_split

train_df=data.drop (['min_ts','M','TS00','TO00','TS01','TO01','TS02','TO02','TS03','TO03','TS04','TO04','TS05','TO05','TS06','TO06','TS07','TO07','TS08','TO08','TS09','TO09','TS10','TO10','TS11','TO11','TS12','TO12','TS13','TO13','TS14','TO14'], axis=1)
y=data.min_ts.values
x=train_df.values
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# ### Model Training: LightGBM Regressor (GBT Boosting)

# In[83]:

import lightgbm

train_data = lightgbm.Dataset(x, label=y)
test_data = lightgbm.Dataset(x_test, label=y_test)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

evals_result = {}
gbm = lightgbm.train(params,
                train_data,
                num_boost_round=200,
                valid_sets=test_data,
                early_stopping_rounds=5,
                evals_result=evals_result,
                feature_name = ['X1','X2','X3'],
                categorical_feature=['X3'])  #treating X3 as categorical


# ### Feature Importance and Eval_metrics plots

# In[84]:

lightgbm.plot_importance(gbm)
plt.show()

lightgbm.plot_metric(evals_result, metric='l1', title='l1 Metric during training')
plt.show()

lightgbm.plot_metric(evals_result, metric='l2', title='l2 Metric during training')
plt.show()

lightgbm.plot_tree(gbm, tree_index=1, figsize=(50, 50), show_info=['split_gain'])
plt.show()


# From the graph we can notice l1 loss to decrease linearly while l2 plateaus after 110 iterations.
# After reaching the minima, l2 starts increasing after iter #140

# ### RMSE: Actual-Predicted

# In[150]:

from sklearn.metrics import mean_squared_error

y_pred = gbm.predict(x_test)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


# Since the *y* value is in seconds, RMSE seems high. On changing *y* to hours, RMSE decreases. 

# ### Writing Test/Eval dataset to disk

# In[145]:

def convertToTime(x):
    hrs=int(x/3600)
    mins=int((x%3600)/60)
    return format(hrs, '02d') +":"+ format(mins, '02d')

output=pd.DataFrame(x_test, columns=['X1','X2','X3'])
output['min_TS']=y_pred
output.min_TS=output.min_TS.apply(convertToTime)


# In[147]:

output.to_csv("submission.csv", index=False)

