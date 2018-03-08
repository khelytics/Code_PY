
############################################################################################################
################################ Event Data Import ###########################################################
############################################################################################################
## Load Libraries 
import pandas as pd
import numpy as np
import datetime as dt

d1 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-b56e2d37-023a-4935-803e-9ccb43f25d3b.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d2 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-00fb46cf-f978-4ac4-a171-8acb3d8c38df.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d3 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-e840147f-6ae5-4970-8db6-e874ff68aed6.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d4 = pd.read_csv("/home/ec2-user/hackathon/NEWpart-r-00000-e3d4e4e6-8ce8-4782-a8c2-6f0727b9dade.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d_hold = pd.read_csv("/home/ec2-user/hackathon/holdOutSampleHeader.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})


edf=pd.concat([d1,d2,d3,d4])

## Convert evt_ts column into Datatime format
edf['evt_ts'] = pd.to_datetime(edf['evt_ts'])
d_hold['evt_ts'] = pd.to_datetime(d_hold['evt_ts'])

### Create new column 
#edf['evt_Year'] = edf['evt_ts'].dt.year
edf['evt_Month'] = edf['evt_ts'].dt.month
d_hold['evt_Month'] = d_hold['evt_ts'].dt.month
#edf['evt_Day'] = edf['evt_ts'].dt.dayofyear
#d_hold['evt_Day'] = d_hold['evt_ts'].dt.dayofyear


### Count of the Unique Act ids
edf['acct_id'].value_counts()
d_hold['acct_id'].value_counts()

##### Create New Acct ID 
edf['New_acct_id']=edf['acct_id'].astype(str)+'_'+edf['evt_Month'].astype(str)
d_hold['New_acct_id']=d_hold['acct_id'].astype(str)+'_'+d_hold['evt_Month'].astype(str)


############################################################################################################
################################ Device Data Import ###########################################################
############################################################################################################

### Import Device Data

df=pd.read_csv('/home/ec2-user/hackathon/Device/part-r-00000-52c9d811-dd02-45f4-9f0e-d67c18e339ba.csv')
### Convert  string to Date format
df['first_login'] = pd.to_datetime(df['first_login'])
df['last_login'] = pd.to_datetime(df['last_login'])


#### Data Type of each column

for col in df:
    print (type(df[col][1]))
#dv_data['first_login'].dtype

##### Create new columns to be used

#df['FL_Year'] = df['first_login'].dt.year
#df['FL_Month'] = df['first_login'].dt.month
#df['FL_Day'] = df['first_login'].dt.dayofyear

#df['LL_Year'] = df['last_login'].dt.year
df['LL_Month'] = df['last_login'].dt.month
#df['LL_Day'] = df['last_login'].dt.dayofyear

### Create new Account_ID
df['New_acct_id']=df['acct_id'].astype(str)+'_'+df['LL_Month'].astype(str)

### Create the count of Logins, DeviceIDs and platform
k = df.groupby("New_acct_id").agg({"n_logins": np.sum, "device_id": pd.Series.nunique,"platform": pd.Series.nunique})

### Create Dataframe from groupby object
k1= k.add_suffix('_Count').reset_index()
k1['platform_Count'].value_counts()

### Check ID Count
len(df)
len(df['New_acct_id'].value_counts())




#############################################################################################################
####################################### Merge Event and Device Data #########################################
#############################################################################################################

Mod_df = pd.merge(edf, k1, how='left', on=['New_acct_id'])
hold_df = pd.merge(d_hold, k1, how='left', on=['New_acct_id'])

###### Create even type Dummy

Mod_dmy = pd.concat([Mod_df, pd.get_dummies(Mod_df['evt_type_cd'])], axis=1);
hold_dmy = pd.concat([hold_df, pd.get_dummies(hold_df['evt_type_cd'])], axis=1);

Mod_dmy.fillna(0)
hold_dmy.fillna(0)


array(['acct_id', 'evt_type_cd', 'evt_ts', 'monetization_ts', 'target',
       'evt_Month', 'New_acct_id', 'n_logins_Count', 'device_id_Count',
       'platform_Count', '012', '013', '020', '024', '042', '043', '173',
       '176', '191', '466', '991', '992', '993', '994'], dtype=object)

mod_data = Mod_dmy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count' ]].copy()
 
hol_data = hold_dmy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count'  ]].copy()

 
mod_data['is_train'] = np.random.uniform(0, 1, len(mod_data)) <= .70

train, test = mod_data[mod_data['is_train']==True], mod_data[mod_data['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print('Number of observations in the holdout data:',len(hol_data))

training_features = [ 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count'  ]
target = 'target'

clf = RandomForestClassifier(n_jobs=2, random_state=0)

trained_model = clf.fit(train[training_features], train[target])

print "Trained model :: ", trained_model
predictions = trained_model.predict(test[training_features])


headers = ["name", "score"]

values = sorted(zip(train[training_features].columns, trained_model.feature_importances_), key=lambda x: x[1] * -1)

print(values, headers)


################### Train and Test Accuracy

print "Train Accuracy :: ", accuracy_score(train[target], trained_model.predict(train[training_features]))
print "Test Accuracy  :: ", accuracy_score(test[target], predictions)


#######################  Holdout Sample

hold_model = clf.fit(hol_data[training_features], hol_data[target])

print "Trained model :: ", hold_model
predictions = hold_model.predict(hol_data[training_features])

print "Hold Accuracy :: ", accuracy_score(hol_data[target], hold_model.predict(hol_data[training_features]))
 
################################ Precision & recall

precision = cross_val_score(clf, train[training_features], train[target], scoring='precision')
recall = cross_val_score(clf, train[training_features], train[target], scoring='recall')




