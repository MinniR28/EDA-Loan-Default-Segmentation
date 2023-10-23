#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[2]:


app = pd.read_csv('application_data.csv')
prev_app = pd.read_csv('previous_application.csv') 


# In[3]:


app.head()


# In[4]:


prev_app.head()


# In[ ]:





# ## Feature Selection

# In[5]:


app.columns


# In[6]:


app.shape


# In[7]:


missing_info =pd.DataFrame(app.isnull().sum().sort_values()).reset_index()
missing_info.rename(columns = {'index':'col_name',0:'null_count'},inplace = True)
missing_info.head()


# In[8]:


# Percentage of missing values

missing_info['msng_pct'] = missing_info['null_count']/app.shape[0]*100
missing_info


# In[ ]:





# In[9]:


msng_col = missing_info[missing_info['msng_pct'] >= 40]['col_name'].to_list()
app_msng_rmvd = app.drop(labels=msng_col,axis=1)
app_msng_rmvd.shape







#app_msng_rmvd = app.drop()


# In[10]:


app_msng_rmvd.head()


# In[11]:


flag_col = []

for col in app_msng_rmvd.columns:
    if col.startswith("FLAG_"):
        flag_col.append(col)
        
flag_col


# In[12]:


flag_tgt_col = app_msng_rmvd[flag_col + ['TARGET']]
flag_tgt_col.head()


# In[13]:


plt.figure(figsize= (20,25))


for i , col in enumerate(flag_col):
    plt.subplot(7,4,i+1)
    sns.countplot(data=flag_tgt_col, x = col , hue = 'TARGET')


# In[14]:


flg_corr = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
 'FLAG_EMAIL','TARGET']
flag_corr_df = app_msng_rmvd[flg_corr]


# In[15]:


flag_corr_df.groupby(['FLAG_OWN_CAR']).size()


# In[16]:


flag_corr_df['FLAG_OWN_CAR'] = flag_corr_df['FLAG_OWN_CAR'].replace({'N':0,'Y':1})
flag_corr_df['FLAG_OWN_REALITY'] = flag_corr_df['FLAG_OWN_REALTY'].replace({'N':0, 'Y':1})

flag_corr_df.groupby(['FLAG_OWN_CAR']).size()


# In[17]:




corr_df = round(flag_corr_df.corr(),2)

plt.figure(figsize= (10,10))
sns.heatmap(corr_df, cmap='coolwarm',linewidths = .5, annot= True)


# In[18]:


app_flag_rmvd = app_msng_rmvd.drop(labels = flag_col,axis =1)
app_flag_rmvd.shape


# In[19]:


app_flag_rmvd.head()


# In[20]:


round(app_flag_rmvd[['EXT_SOURCE_2', 'EXT_SOURCE_3','TARGET']].corr(),2)


# In[21]:


sns.heatmap(round(app_flag_rmvd[['EXT_SOURCE_2', 'EXT_SOURCE_3','TARGET']].corr(),2), cmap='coolwarm', linewidth =.5,annot=True)


# In[22]:


app_score_col_rmvd = app_flag_rmvd.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'],axis=1)


# In[23]:


app_score_col_rmvd.shape


# In[ ]:





# ## Feature Engineering

# In[24]:


app_score_col_rmvd.isnull().sum().sort_values()/app_score_col_rmvd.shape[0]


# ### Missing Imputation

# In[25]:


app_score_col_rmvd.groupby('CNT_FAM_MEMBERS').size()


# In[26]:


app_score_col_rmvd['CNT_FAM_MEMBERS'] = app_score_col_rmvd['CNT_FAM_MEMBERS'].fillna((app_score_col_rmvd['CNT_FAM_MEMBERS'].mode()[0]))


# In[27]:


app_score_col_rmvd['CNT_FAM_MEMBERS'].isnull().sum()


# In[28]:


app_score_col_rmvd.groupby(['OCCUPATION_TYPE']).size().sort_values()


# In[29]:


app_score_col_rmvd['OCCUPATION_TYPE'] = app_score_col_rmvd['OCCUPATION_TYPE'].fillna((app_score_col_rmvd['OCCUPATION_TYPE'].mode()[0]))

#app_score_col_rmvd['OCCUPATION_TYPE'].mode()


# In[30]:


app_score_col_rmvd['OCCUPATION_TYPE'].isnull().sum()


# In[31]:


app_score_col_rmvd['NAME_TYPE_SUITE'].isnull().sum()


# In[32]:


app_score_col_rmvd['NAME_TYPE_SUITE'].mode()


# In[33]:


app_score_col_rmvd.groupby(['NAME_TYPE_SUITE']).size()


# In[34]:


app_score_col_rmvd['NAME_TYPE_SUITE'] = app_score_col_rmvd['NAME_TYPE_SUITE'].fillna((app_score_col_rmvd['NAME_TYPE_SUITE'].mode()[0]))


# In[35]:


app_score_col_rmvd['NAME_TYPE_SUITE'].isnull().sum()


# In[36]:


app_score_col_rmvd['AMT_ANNUITY'].describe()


# In[37]:


app_score_col_rmvd['AMT_ANNUITY'] = app_score_col_rmvd['AMT_ANNUITY'].fillna((app_score_col_rmvd['AMT_ANNUITY'].mean()))

                                                                             


# In[38]:


app_score_col_rmvd['AMT_ANNUITY'].isnull().sum()


# In[39]:


app_score_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].describe()


# In[40]:


app_score_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].head()


# In[41]:


amt_req_col = []

for col in app_score_col_rmvd.columns:
    if col.startswith('AMT_REQ_CREDIT_BUREAU_HOUR'):
        amt_req_col.append(col)
        
amt_req_col

#app_score_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].unique()


# In[42]:


for col in amt_req_col:
    app_score_col_rmvd[col] = app_score_col_rmvd['AMT_ANNUITY'].fillna((app_score_col_rmvd[col].median()))


# In[43]:


app_score_col_rmvd[col].isnull().sum()


# In[44]:


app_score_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[45]:


app_score_col_rmvd['AMT_GOODS_PRICE'].describe()


# In[46]:


app_score_col_rmvd['AMT_GOODS_PRICE'].agg(['min','max','median'])


# In[47]:


app_score_col_rmvd['AMT_GOODS_PRICE'].mean()


# In[48]:


app_score_col_rmvd['AMT_GOODS_PRICE'].median()


# In[49]:


app_score_col_rmvd['AMT_GOODS_PRICE'] = app_score_col_rmvd['AMT_GOODS_PRICE'].fillna((app_score_col_rmvd['AMT_GOODS_PRICE'].median()))


# In[50]:


app_score_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[51]:


app_score_col_rmvd.head()


# In[52]:


app_score_col_rmvd['AMT_CREDIT'].isnull().sum()


# In[53]:


days_col = []

for col in app_score_col_rmvd.columns:
    if col.startswith("DAYS"):
        days_col.append(col)
        
days_col


# In[54]:


# Converting the negative vaues to positive

for col in days_col:
    app_score_col_rmvd[col] = abs(app_score_col_rmvd[col])


# In[55]:


app_score_col_rmvd.head()


# In[56]:


app_score_col_rmvd.info()


# In[57]:


app_score_col_rmvd.nunique()


# In[58]:


app_score_col_rmvd.nunique().sort_values()


# In[59]:


app_score_col_rmvd['OBS_30_CNT_SOCIAL_CIRCLE'].unique()


# ## Outlier detection & Treatment

# In[60]:


app_score_col_rmvd['AMT_GOODS_PRICE'].agg(['min','max','median'])


# In[61]:


sns.kdeplot(data = app_score_col_rmvd, x= 'AMT_GOODS_PRICE')


# In[62]:


sns.boxenplot(data = app_score_col_rmvd, x= 'AMT_GOODS_PRICE')


# In[63]:


app_score_col_rmvd['AMT_GOODS_PRICE'].quantile([0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7, 0.8, 0.9, 0.99])


# In[64]:


bins = [0,100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 4050000]
ranges = ['0-100k','100k-200k','200k-300k','300k-400k','400k-500k','500k-600k','600k-700k',
          '700k-800k','800k-900k','Above 900k']


app_score_col_rmvd['AMT_GOODS_PRICE_RANGE'] = pd.cut(app_score_col_rmvd['AMT_GOODS_PRICE'],bins, labels= ranges)


# In[65]:


app_score_col_rmvd.groupby(['AMT_GOODS_PRICE_RANGE']).size()


# In[66]:


app_score_col_rmvd['AMT_INCOME_TOTAL'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[67]:


app_score_col_rmvd['AMT_INCOME_TOTAL'].max()


# In[68]:


bins = [0,100000,150000,200000,250000,300000,350000,400000,117000000]
ranges = ['0-100K','100K-150K','150K-200K','200K-250K','250K-300K','300K-350K','350K-400K'
          ,'Above 400K']


# In[69]:


app_score_col_rmvd['AMT_INCOME_TOTAL_RANGE'] = pd.cut(app_score_col_rmvd['AMT_INCOME_TOTAL'],bins,labels=ranges)


# In[70]:


app_score_col_rmvd.groupby(['AMT_INCOME_TOTAL_RANGE']).size()


# In[71]:


app_score_col_rmvd['AMT_INCOME_TOTAL_RANGE'].isnull().sum()


# In[72]:


app_score_col_rmvd['AMT_CREDIT'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[73]:


app_score_col_rmvd['AMT_CREDIT'].max()


# In[74]:


bins = [0,200000,400000,600000,800000,900000,1000000,2000000,3000000,4050000]
ranges = ['0-200K','200K-400K','400K-600K','600K-800K','800K-900K','900K-1M','1M-2M','2M-3M','Above 3M']

app_score_col_rmvd['AMT_CREDIT_RANGE'] = pd.cut(app_score_col_rmvd['AMT_CREDIT'],bins,labels=ranges)


# In[75]:


app_score_col_rmvd.groupby(['AMT_CREDIT_RANGE']).size()


# In[76]:


app_score_col_rmvd['AMT_CREDIT'].isnull().sum()


# In[77]:


app_score_col_rmvd['AMT_ANNUITY'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[78]:


app_score_col_rmvd['AMT_ANNUITY'].max()


# In[79]:


bins = [0,25000,50000,100000,150000,200000,258025.5]
ranges = ['0-25K','25K-50K','50K-100K','100K-150K','150K-200K','Above 200K']

app_score_col_rmvd['AMT_ANNUITY_RANGE'] = pd.cut(app_score_col_rmvd['AMT_ANNUITY'],bins,labels=ranges)


# In[80]:


app_score_col_rmvd.groupby(['AMT_ANNUITY_RANGE']).size()


# In[81]:


app_score_col_rmvd['AMT_ANNUITY_RANGE'].isnull().sum()


# In[82]:


app_score_col_rmvd['DAYS_EMPLOYED'].agg(['min','max','median'])


# In[83]:


app_score_col_rmvd['DAYS_EMPLOYED'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[84]:


app_score_col_rmvd[app_score_col_rmvd['DAYS_EMPLOYED']<app_score_col_rmvd['DAYS_EMPLOYED'].max()].max()['DAYS_EMPLOYED']


# In[85]:


app_score_col_rmvd['DAYS_EMPLOYED'].max()


# In[86]:


app_score_col_rmvd['DAYS_EMPLOYED'].max()


# In[87]:


bins = [0,1825,3650,5475,7300,9125,10950,12775,14600,16425,18250,23691,365243]

ranges = ['0-5Y','5Y-10Y','10Y-15Y','15Y-20Y','20Y-25Y','25Y-30Y','30Y-35Y','35Y-40Y','40Y-45Y','45Y-50Y'
          ,'50Y-65Y','Above 65Y']

app_score_col_rmvd['DAYS_EMPLOYED_RANGE'] = pd.cut(app_score_col_rmvd['DAYS_EMPLOYED'],bins,labels=ranges)


# In[88]:


app_score_col_rmvd.groupby(['DAYS_EMPLOYED_RANGE']).size()


# In[89]:


app_score_col_rmvd['DAYS_BIRTH'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[90]:


bins = [0,7300,10950,14600,18250,21900,25229]

ranges = ['20Y','20Y-30Y','30Y-40Y','40Y-50Y','50Y-60Y','Above 60Y']

app_score_col_rmvd['DAYS_BIRTH_RANGE'] = pd.cut(app_score_col_rmvd['DAYS_BIRTH'],bins,labels=ranges)


# In[91]:


app_score_col_rmvd.groupby(['DAYS_BIRTH_RANGE']).size()


# In[92]:


app_score_col_rmvd['DAYS_BIRTH'].isnull().sum()


# In[93]:


app_score_col_rmvd.dtypes.value_counts()


# In[ ]:





# ## Data Analysis

# In[94]:


app_score_col_rmvd.info()


# In[95]:


app_score_col_rmvd.dtypes.value_counts()


# In[96]:


# Info of the column where the data type is object


app_score_col_rmvd.select_dtypes(include=['object']).head()


# In[97]:


obj_var = app_score_col_rmvd.select_dtypes(include=['object']).columns


# In[98]:


obj_var


# In[99]:


app_score_col_rmvd.groupby(['NAME_CONTRACT_TYPE']).size()


# In[100]:


sns.countplot(data=app_score_col_rmvd, x ='NAME_CONTRACT_TYPE', hue = 'TARGET')


# In[101]:


data_pct = app_score_col_rmvd[['NAME_CONTRACT_TYPE','TARGET']].groupby(['NAME_CONTRACT_TYPE'], as_index=False).mean().sort_values(by='TARGET',ascending=False)


# In[102]:


data_pct


# In[103]:


data_pct['PCT'] = data_pct['TARGET']*100


# In[104]:


data_pct


# In[105]:


sns.barplot(data=data_pct,x='NAME_CONTRACT_TYPE',y='PCT')


# In[106]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(data=app_score_col_rmvd,x='NAME_CONTRACT_TYPE',hue='TARGET')

plt.subplot(1,2,2)
sns.barplot(data=data_pct,x='NAME_CONTRACT_TYPE',y='PCT')


# In[107]:


obj_var


# In[108]:


plt.figure(figsize=(25,60))


for i, var in enumerate(obj_var):

    data_pct = app_score_col_rmvd[[var,'TARGET']].groupby([var], as_index=False).mean().sort_values(by='TARGET',ascending=False)
    data_pct['PCT'] = data_pct['TARGET']*100


    plt.subplot(10,2,i+i+1)
    plt.subplots_adjust(wspace=0.1,hspace=1)
    sns.countplot(data=app_score_col_rmvd,x=var,hue='TARGET')
    plt.xticks(rotation=90)

    plt.subplot(10,2,i+i+2)
    sns.barplot(data=data_pct,x=var,y='PCT',palette='coolwarm')
    plt.xticks(rotation=90)


# In[ ]:





# In[109]:


app_score_col_rmvd['NAME_EDUCATION_TYPE'].unique()


# In[110]:


app_score_col_rmvd.dtypes.value_counts()


# In[111]:


num_var = app_score_col_rmvd.select_dtypes(include = ['float64','int64']).columns
num_cat_var = app_score_col_rmvd.select_dtypes(include=['float64','int64','category']).columns
len(num_var)


# In[112]:


num_data = app_score_col_rmvd[num_var]
defaulters = num_data[num_data['TARGET']==1]
repayers = num_data[num_data['TARGET']==0]
repayers.head()


# In[113]:


defaulters.head()


# In[114]:


defaulters[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL']].corr()


# In[115]:


defaulter_corr = defaulters.corr()
defaulter_corr


# In[116]:


defaulter_corr = defaulters.corr()

defaulter_corr_unstck = defaulter_corr.where(np.triu(np.ones(defaulter_corr.shape),k=1).astype(np.bool)).unstack().reset_index().rename(columns={'level_0': 'var1',
                                                                                                                        'level_1': 'var2',
                                                                                                                        0:'corr'})
defaulter_corr_unstck.head()


# In[117]:


defaulter_corr = defaulters.corr()

defaulter_corr_unstck = defaulter_corr.where(np.triu(np.ones(defaulter_corr.shape),k=1).astype(np.bool)).unstack().reset_index().rename(columns={'level_0': 'var1',
                                                                                                                        'level_1': 'var2',
                                                                                                                        0:'corr'})
defaulter_corr_unstck['corr'] = abs(defaulter_corr_unstck['corr'])
defaulter_corr_unstck.head()


# In[118]:


repayers_corr = repayers.corr()
repayers_corr_unstck = repayers_corr.where(np.triu(np.ones(repayers_corr.shape),k=1).astype(np.bool)).unstack().reset_index().rename(columns={'level_0':'var1'
                                                                                                                        ,'level_1':'var2'
                                                                                                                        ,0:'corr'})

repayers_corr_unstck['corr'] = abs(repayers_corr_unstck['corr'])
repayers_corr_unstck.dropna(subset=['corr']).sort_values(by=['corr'],ascending=False).head(10)


# In[119]:


# Univariate Analysis

num_data.head()


# In[120]:


amt_var = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']


# In[121]:



sns.kdeplot(data=num_data,x='AMT_CREDIT',hue='TARGET')


# In[122]:


plt.figure(figsize=(10,5))

for i, col in enumerate(amt_var):
    plt.subplot(2,2,i+1)
    sns.kdeplot(data=num_data,x=col,hue='TARGET')
    plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[123]:


num_data.head()


# In[124]:


# Bivariate Analysis

sns.scatterplot(data=num_data,x='AMT_CREDIT',y='CNT_CHILDREN',hue='TARGET')


# In[125]:


sns.scatterplot(data=num_data, x='AMT_CREDIT', y='AMT_INCOME_TOTAL',hue='TARGET')


# In[126]:


# AMT_CREDIT and AMT_GOODS_PRICE are linearly correlated 

sns.scatterplot(data=num_data,x='AMT_CREDIT',y='AMT_GOODS_PRICE',hue='TARGET')


# In[127]:


amt_var = num_data[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','TARGET']]


# In[128]:


sns.pairplot(data=amt_var,hue='TARGET')


# In[129]:


null_count = pd.DataFrame(prev_app.isnull().sum().sort_values(ascending=False)/prev_app.shape[0]*100).reset_index().rename(columns={'index':'var',
                                                                                                                                   0:'count_pct'})
var_msng_ge_40 = list(null_count[null_count['count_pct']>=40]['var'])
var_msng_ge_40


# In[130]:


nva_cols = var_msng_ge_40+['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']
len(nva_cols)


# In[131]:


len(prev_app.columns)


# In[132]:


prev_app_nva_col_rmvd = prev_app.drop(labels=nva_cols,axis=1)


len(prev_app_nva_col_rmvd.columns)


# In[133]:


prev_app_nva_col_rmvd.columns


# In[134]:


prev_app_nva_col_rmvd.head()


# In[135]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)/prev_app_nva_col_rmvd.shape[0]*100


# In[136]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].agg(func=['mean','median'])


# In[137]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MEDIAN'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].median())


# In[138]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MEAN'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].mean())


# In[139]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE_MODE'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].mode()[0])


# In[140]:


gp_cols = ['AMT_GOODS_PRICE','AMT_GOODS_PRICE_MEDIAN','AMT_GOODS_PRICE_MEAN','AMT_GOODS_PRICE_MODE']


# In[141]:


plt.figure(figsize=(10,5))

for i, col in enumerate(gp_cols):
    plt.subplot(2,2,i+1)
    sns.kdeplot(data=prev_app_nva_col_rmvd,x=col)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[142]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'] = prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].fillna(prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].median())


# In[143]:


prev_app_nva_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[145]:


prev_app_nva_col_rmvd['AMT_ANNUITY'].agg(func=['mean','median','max'])


# In[146]:


prev_app_nva_col_rmvd['AMT_ANNUITY'] = prev_app_nva_col_rmvd['AMT_ANNUITY'].fillna(prev_app_nva_col_rmvd['AMT_ANNUITY'].median())


# In[147]:


prev_app_nva_col_rmvd['PRODUCT_COMBINATION'] = prev_app_nva_col_rmvd['PRODUCT_COMBINATION'].fillna(prev_app_nva_col_rmvd['PRODUCT_COMBINATION'].mode()[0])


# In[148]:


prev_app_nva_col_rmvd['CNT_PAYMENT'].agg(func=['mean','median','max'])


# In[149]:


prev_app_nva_col_rmvd[prev_app_nva_col_rmvd['CNT_PAYMENT'].isnull()].groupby(['NAME_CONTRACT_STATUS']).size().sort_values(ascending=False)


# In[150]:


prev_app_nva_col_rmvd['CNT_PAYMENT'] = prev_app_nva_col_rmvd['CNT_PAYMENT'].fillna(0)


# In[151]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)


# In[152]:


prev_app_nva_col_rmvd = prev_app_nva_col_rmvd.drop(labels=['AMT_GOODS_PRICE_MEDIAN','AMT_GOODS_PRICE_MEAN','AMT_GOODS_PRICE_MODE'],axis=1)


# In[153]:


prev_app_nva_col_rmvd.isnull().sum().sort_values(ascending=False)


# In[154]:


len(prev_app_nva_col_rmvd.columns)


# In[155]:


prev_app_nva_col_rmvd.head()


# In[156]:


merged_df = pd.merge(app_score_col_rmvd,prev_app_nva_col_rmvd,how='inner',on='SK_ID_CURR')
merged_df.head()


# In[157]:


plt.figure(figsize=(15,5))

sns.countplot(data=merged_df,x='NAME_CASH_LOAN_PURPOSE',hue='NAME_CONTRACT_STATUS')
plt.xticks(rotation=90)
plt.yscale('log')


# In[158]:


sns.countplot(data=merged_df,x='NAME_CONTRACT_STATUS',hue='TARGET')


# In[159]:


merged_agg = merged_df.groupby(['NAME_CONTRACT_STATUS','TARGET']).size().reset_index().rename(columns={0:'counts'})
sum_df  = merged_agg.groupby(['NAME_CONTRACT_STATUS'])['counts'].sum().reset_index()

merged_agg_2 = pd.merge(merged_agg,sum_df,how='left',on='NAME_CONTRACT_STATUS')
merged_agg_2['pct'] = round(merged_agg_2['counts_x']/merged_agg_2['counts_y']*100,2)
merged_agg_2


# In[160]:


sns.lineplot(data=merged_df,x='NAME_CONTRACT_STATUS',y='AMT_INCOME_TOTAL',ci=None,hue='TARGET')


# In[161]:



len(merged_df.columns)


# In[162]:


# # All the analysis

# most of the customers have taken cash loan
# customers who have taken cash loans are less likely to default
# 
# CODE_GENDER - 
# 
#     most of the loans have been taken by female
#     default rate for females are just ~7% which is safer and lesser than male
# 
# NAME_TYPE_SUITE - 
# 
#     unacompanied people had tanke most of the loans and the default rate is ~8.5% which is still okay
# 
# NAME_INCOME_TYPE - 
# 
#     the safest segments are working, commercial associates and pensioners
# 
# NAME_EDUCATION_TYPE - 
# 
#     Higher education is the safest segment to give the loan with a default rate of less than 5%
# 
# NAME_FAMILY_STATUS - 
# 
#     Married people are safe to target, default rate is 8%
# 
# 
# NAME_HOUSING_TYPE - 
# 
#     People having house/appartment are safe to give the loan with default rate of ~8%
# 
# OCCUPATION_TYPE - 
# 
#     Low-Skill Laboreres and drivers are highest defaulters
#     Accountants are less defaulters
#     Core staff, Managers and Laborers are safer to target with a default rate of <= 7.5 to 10%
# 
# ORGANIZATION_TYPE - 
# 
#     Transport type 3 highest defaulter
#     Others, Business Entity Type 3, Self Employed are good to go with default rate around 10 %
# 
# =======univariate numeric variables analysis========
# 
#     >> most of the loans were given for the goods price ranging between 0 to 1 ml
#     >> most of the loans were given for the credit amount of 0 to 1 ml
#     >> most of the customers are paying annuity of 0 to 50 K
#     >> mostly the customers have income between 0 to 1 ml
# 
# =============bivariate analysis==================
# 
#     >> AMT_CREDIT and AMT_GOODS_PRICE are linearly corelated, if the AMT_CREDIT increases the defaulters are decreasing
#     >> people having income less than or equals to 1 ml, are more like to take loans out of which who are taking loan of less than 1.5 million, coudl turn out to be defaulters. we can target income below 1 million and loan maount greater than 1.5 million
#     >> people having children 1 to less than 5 are safer to give the loan
#     >> People who can pay the annuity of 100K are more like to get the loan and that's upto less than 2ml (safer segment)
# 
# ============analysis on merged data==============
# 
#     >> for the repairing purpose customers had applied mostly prev. and the same puspose has most number of cancelations
#     >> most of the app. which were prev. either canceled or refused 80-90% of them are repayer in the current data
#     >> offers which were unused prev. now have maximum number of defaulters despite of having high income band customers

# # Final Conclusion/Insights

# Bank should target the customers
# 
#     >> having low income i.e. below 1 ml
#     >> working in Others, Business Entity Type 3, Self Employed  org. type
#     >> working as Accountants, Core staff, Managers and Laborers 
#     >> having house/appartment and are married and having children not more than 5
#     >> Highly educated
#     >> preferably female
# 
#     >> unacompanied people can be safer -  default rate is ~8.5%
# 
# Amount segment recommended -
# 
#     >> the credit amount should not be more than 1 ml
#     >> annuity can be made of 50K (depending on the eligibility)
#     >> income bracket could be below 1 ml
# 
#     >> 80-90% of the customer who were prev. canceled/refused, are repayers. Bank can do the analysis and can consider to give loan to these segments
# 
# 
# ====================precautions===============
# 
#     >> org. Transport type 3 should be avoided
#     >> Low-Skill Laboreres and drivers  should be avoided
#     >> offers prev. unused and high income customer should be avoided
# 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Value Modification

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




