
# coding: utf-8

# # Import Libraries and Load datasets

# In[1]:


# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import datetime as dt

get_ipython().run_line_magic('matplotlib', 'inline')
seed = 129

# Import Data

path = 'input/'

transactions = pd.read_csv(path+'transactions_v2.csv',parse_dates=['transaction_date','membership_expire_date'], dtype={'payment_method_id':np.int8,'payment_plan_days':np.int8,'plan_list_price':np.int8,'actual_amount_paid':np.int8,'is_auto_renew':np.int8,'is_cancel':np.int8})
train = pd.read_csv(path+'train_v2.csv',dtype={'is_churn':np.int8})

test = pd.read_csv(path+'sample_submission_v2.csv',dtype={'is_churn':np.int8})

members = pd.read_csv(path+'members_v3.csv',parse_dates=['registration_init_time'],dtype={'city':np.int8,'bd':np.int8,'registered_via':np.int8})



user_log = pd.read_csv(path+'user_logs_v2.csv',parse_dates=['date'],dtype={'num_25':np.int16,'num_50':np.int16, 'num_75':np.int16,'num_985':np.int16,'num_100':np.int16,'num_unq':np.int16})



# # User Log dataset info

# In[2]:


print('User Log Data : Number of rows & columns',user_log.shape)
user_log.head()


# # Transactions dataset info

# In[3]:


print('Transactions Data : Number of rows & columns',transactions.shape)
transactions.head()


# # Members dataset info

# In[4]:


print('Members Data : Number of rows & columns',members.shape)
members.head()


# # Test dataset info

# In[5]:


print('Test Data : Number of rows & columns',test.shape)
test.head()


# # Train dataset info

# In[6]:


print('Train Data : Number of rows & columns',train.shape)
train.head()


# # All dataset column descrption

# In[7]:


print('\nTrain:',train.describe().T)
print('\nTest:',test.describe().T)
print('\nMembers:',members.describe().T)
print('\nTransactions:',transactions.describe().T)
print('\nUser log:',user_log.describe().T)


# # Merge Train & Test dataset with Members, Trasanctions & User Logs

# In[8]:


train = pd.merge(train,members,on='msno',how='left')
test = pd.merge(test,members,on='msno',how='left')
train = pd.merge(train,transactions,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,transactions,how='left',on='msno',left_index=True, right_index=True,)
train = pd.merge(train,user_log,how='left',on='msno',left_index=True, right_index=True)
test = pd.merge(test,user_log,how='left',on='msno',left_index=True, right_index=True)

del members,transactions,user_log
print('Number of rows & columns',train.shape)
print('Number of rows & columns',test.shape)


# # Generate descriptive statistics on date columns

# In[9]:


train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].describe()


# # Find null value count for date columns

# In[10]:


train[['registration_init_time' ,'transaction_date','membership_expire_date','date']].isnull().sum()


# # Update null values with default values - registration_init_time

# In[11]:


train['registration_init_time'] = train['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))
test['registration_init_time'] = test['registration_init_time'].fillna(value=pd.to_datetime('09/10/2015'))


# # Feature Genration :- Create numerical columns of date columns

# In[12]:


def date_feature(df):
    
    col = ['registration_init_time' ,'transaction_date','membership_expire_date','date']
    var = ['reg','trans','mem_exp','user_']
    #df['duration'] = (df[col[1]] - df[col[0]]).dt.days 
    
    for i ,j in zip(col,var):
        df[j+'_day'] = df[i].dt.day.astype('uint8')
        df[j+'_weekday'] = df[i].dt.weekday.astype('uint8')        
        df[j+'_month'] = df[i].dt.month.astype('uint8') 
        df[j+'_year'] =df[i].dt.year.astype('uint16') 

date_feature(train)
date_feature(test)


# # Find null value columns in numerical columns

# In[13]:


train.isnull().sum()


# # Update null values with default values

# In[14]:


col = [ 'city', 'bd', 'gender', 'registered_via']
def missing(df,columns):
    col = columns
    for i in col:
        df[i].fillna(df[i].mode()[0],inplace=True)

missing(train,col)
missing(test,col)


# # Label encoder - Gender column

# In[15]:


# Encode Label to vlaues 0 or 1. That is male, female to 0-1.
le = LabelEncoder()
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])


# # Feature Generation - discount, is_discount, membership_duration & registration_duration

# In[16]:


train['discount'] = train['plan_list_price'] - train['actual_amount_paid']
test['discount'] = test['plan_list_price'] - test['actual_amount_paid']

train['is_discount'] = train.discount.apply(lambda x: 1 if x > 0 else 0)
test['is_discount'] = test.discount.apply(lambda x: 1 if x > 0 else 0)



# In[17]:


train['membership_duration'] = train.membership_expire_date - train.transaction_date
train['membership_duration'] = train['membership_duration'] / np.timedelta64(1, 'D')
train['membership_duration'] = train['membership_duration'].astype(int)

test['membership_duration'] = test.membership_expire_date - test.transaction_date
test['membership_duration'] = test['membership_duration'] / np.timedelta64(1, 'D')
test['membership_duration'] = test['membership_duration'].astype(int)


# In[18]:


train['registration_duration'] = train.membership_expire_date - train.registration_init_time
train['registration_duration'] = train['registration_duration'] / np.timedelta64(1, 'D')
train['registration_duration'] = train['registration_duration'].astype(int)

test['registration_duration'] = test.membership_expire_date - test.registration_init_time
test['registration_duration'] = test['registration_duration'] / np.timedelta64(1, 'D')
test['registration_duration'] = test['registration_duration'].astype(int)


# In[19]:


train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)

test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)


# In[20]:


train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)

test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)


# # is_Churn

# In[21]:


plt.figure(figsize=(8,6))
sns.set_style('ticks')
sns.countplot(train['is_churn'],palette='summer')
plt.xlabel('The subscription within 30 days of expiration is True/False')


# # Data Analysis

# In[22]:


print(train['city'].unique())
fig,ax = plt.subplots(2,2,figsize=(16,8))
ax1,ax2,ax3,ax4 = ax.flatten()

sns.set(style="ticks")
sns.countplot(train['city'],palette='summer',ax=ax1)
#ax1.set_yscale('log')

ax1.set_xlabel('City')
#ax1.set_xticks(rotation=45)

sns.countplot(x='gender',data = train,palette='winter',ax=ax2)
#ax2.set_yscale('log')
ax2.set_xlabel('Gender')

sns.countplot(x='registered_via',data=train,palette='winter',ax=ax3)
#ax3.set_yscale('')
ax3.set_xlabel('Register via')

sns.countplot(x='payment_method_id',data= train,palette='winter',ax=ax4)
ax4.set_xlabel('Payment_method_id')


# # Gender : 0 - Femal, 1 - Male

# In[23]:


print(pd.crosstab(train['is_churn'],train['gender']))


# # Registration init time

# In[24]:


regi = train.groupby('registration_init_time').count()['is_churn']
plt.subplot(211)
plt.plot(regi,color='b',label='count')
plt.legend(loc='center')
regi = train.groupby('registration_init_time').mean()['is_churn']
plt.subplot(212)
plt.plot(regi,color='r',label='mean')
plt.legend(loc='center')
plt.tight_layout()


# #  Registration

# In[25]:


fig,ax = plt.subplots(2,2,figsize=(16,8))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.countplot(train['reg_day'],palette='Set2',ax=ax1)
sns.countplot(data=train,x='reg_month',palette='Set1',ax=ax2)
sns.countplot(data=train,x='reg_year',palette='magma',ax=ax3)

