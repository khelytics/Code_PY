
############################################################################################################
################################ Event Data Import ###########################################################
############################################################################################################
## Load Libraries 
import pandas as pd
import numpy as np
import datetime as dt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

d1 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-b56e2d37-023a-4935-803e-9ccb43f25d3b.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d2 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-00fb46cf-f978-4ac4-a171-8acb3d8c38df.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d3 = pd.read_csv("/home/ec2-user/hackathon/part-r-00000-e840147f-6ae5-4970-8db6-e874ff68aed6.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d4 = pd.read_csv("/home/ec2-user/hackathon/NEWpart-r-00000-e3d4e4e6-8ce8-4782-a8c2-6f0727b9dade.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d_hold = pd.read_csv("/home/ec2-user/hackathon/holdOutSampleHeader.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})


edf=pd.concat([d1,d2,d3,d4])
lst = [d1,d2,d3,d4]

### Delete unwanted data
del d1
del d2
del d3
del d4

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

######## Create the count of account IDs by Device

dev_id = df.groupby("device_id").agg({ "acct_id": pd.Series.nunique})

### Create data frame from the object
dev_id_df= dev_id.add_suffix('_Count').reset_index()
df = pd.merge(df, dev_id_df, how='left', on=['device_id'])
### delete unwanted data
del dev_id_df

#### Data Type of each column

#for col in df:
#    print (type(df[col][1]))
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
k = df.groupby("New_acct_id").agg({"n_logins": np.sum,"acct_id_Count": np.sum, "device_id": pd.Series.nunique,"platform": pd.Series.nunique})
### Delete the data df
del df
### Create Dataframe from groupby object
k1= k.add_suffix('_Count').reset_index()
#### Delete unused data 
del k
#k1['platform_Count'].value_counts()


#############################################################################################################
####################################### Merge Event and Device Data #########################################
#############################################################################################################

Mod_df = pd.merge(edf, k1, how='left', on=['New_acct_id'])
hold_df = pd.merge(d_hold, k1, how='left', on=['New_acct_id'])

#### Delete unwanted Data
del edf
del d_hold
del k1



#####################################################################################################
#########################      Load Speak easy Data    ############################################
#####################################################################################################

###### Load ATO Speak Easy Data
se_ato = []  # a list of all objects

with open('/home/ec2-user/hackathon/SpeakEasy/ato_speakeasy_transcripts.json') as ato_data:
  for line in ato_data:
    se_ato.append( json.loads(line) ) 

ato_df=pd.DataFrame(se_ato)

################ Find the Key words in the Speak easy

ato_df['expedite'] = np.where(
    ato_df.transcript.str.contains('(?<=unidentified).*?expedite.*?(?=agent)'), '1', '0'
)

ato_df['fedex'] = np.where(
    ato_df.transcript.str.contains('(?<=unidentified).*?fedex.*?(?=agent)'), '1', '0'
)

ato_df['quick'] = np.where(
    ato_df.transcript.str.contains('(?<=unidentified).*?quick.*?(?=agent)'), '1', '0'
)

ato_df['fast'] = np.where(
    ato_df.transcript.str.contains('(?<=unidentified).*?fast.*?(?=agent)'), '1', '0'
)

del se_ato

########################################################################################################
###################### Import Mapping for ATO SE And Assign Account IDs #######################
#######################################################################################################

ato_map = pd.read_csv("/home/ec2-user/hackathon/SpeakEasy/ATO_icm_contact_id_TO_acct_id_mapping.csv")	
#ato_map
#ato_map.columns.values
#array(['CNTCT_ID', 'SEG_ACCT_ID'], dtype=object)
### Rename the Contact ID
ato_map.rename(columns={'CNTCT_ID':'icm_contact_id'}, inplace=True)
#len(ato_map)
#892569
### Remove Duplicates
ato_map = ato_map.drop_duplicates(subset=['icm_contact_id','SEG_ACCT_ID'])
 #len(ato_map)
#576742
#### Change the Column data Type 
#ato_map['icm_contact_id'].apply(str)
ato_df['icm_contact_id']=ato_df['icm_contact_id'].astype(int)
#ato_df['icm_contact_id']

#### Count of Data  and mapping file
#ato_map[ato_map['icm_contact_id']==1522440000333974]
#ato_df[ato_df['icm_contact_id']==1522440000333974]
#len(ato_map['icm_contact_id'].value_counts())
#len(ato_df['icm_contact_id'].value_counts())

###### Asssign Account IDs to the data(Merging)
ato_mapped_data= pd.merge(ato_df, ato_map, on=['icm_contact_id'])



#nato_data
### Delete extra column
del ato_mapped_data['tag']
se_ato_nato=ato_mapped_data

###### Convert string  to dayetime
se_ato_nato['start_time'] = pd.to_datetime(se_ato_nato['start_time'])
#### Craete new var with day
se_ato_nato['dayofyear'] = se_ato_nato['start_time'].dt.dayofyear
#### Creating matching variable with accont ID & Day
se_ato_nato['Day_acct_id']=se_ato_nato['SEG_ACCT_ID'].astype(str)+'_'+se_ato_nato['dayofyear'].astype(str)

###### Selecting useful columns
se_ato_nato_1 = se_ato_nato[['Day_acct_id','expedite','fedex','quick','fast']]

del se_ato_nato
#del nato_df
del ato_df
del ato_mapped_data
#del nato_mapped_data


##### Rem Duplicates

se_ato_nato_1 = se_ato_nato_1.drop_duplicates(subset=['Day_acct_id','expedite','fedex','quick','fast'])
######################################################################################################
######################## Merge SpeakEasy Data with Event Data ####################################
#####################################################################################################

Mod_df['evt_Day'] = Mod_df['evt_ts'].dt.dayofyear
hold_df['evt_Day'] = hold_df['evt_ts'].dt.dayofyear
Mod_df['Day_acct_id']=Mod_df['acct_id'].astype(str)+'_'+Mod_df['evt_Day'].astype(str)
hold_df['Day_acct_id']=hold_df['acct_id'].astype(str)+'_'+hold_df['evt_Day'].astype(str)



######## Merging Speak easy with Event Data
Model_data = pd.merge(Mod_df, se_ato_nato_1, how='left', on=['Day_acct_id'])
Mhold_data = pd.merge(hold_df, se_ato_nato_1, how='left', on=['Day_acct_id'])




###### Create even type Dummy

Mod_dmy = pd.concat([Model_data, pd.get_dummies(Model_data['evt_type_cd'])], axis=1);
hold_dmy = pd.concat([Mhold_data, pd.get_dummies(Mhold_data['evt_type_cd'])], axis=1);

Model_data['expedite'].unique
#Mod_dmy.fillna(0)
#hold_dmy.fillna(0)


'''
array(['acct_id', 'evt_type_cd', 'evt_ts', 'monetization_ts', 'target',
       'evt_Month', 'New_acct_id', 'n_logins_Count', 'device_id_Count',
       'platform_Count', '012', '013', '020', '024', '042', '043', '173',
       '176', '191', '466', '991', '992', '993', '994'], dtype=object)
'''

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
 'acct_id_Count_Count',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count',
 'expedite',
 'fast',
 'quick']].copy()
 
 
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
 'acct_id_Count_Count',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count',
 'expedite',
 'fast',
 'quick' ]].copy()

dt=mod_data.fillna(0)
ht=hol_data.fillna(0)
 
mod_data['is_train'] = np.random.uniform(0, 1, len(mod_data)) <= .70

train, test = dt[dt['is_train']==True], dt[dt['is_train']==False]

#train.fillna(0)
#test.fillna(0)

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print('Number of observations in the holdout data:',len(hol_data))

#train['acct_id_Count_Count']

training_features = [ '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'acct_id_Count_Count',
 'n_logins_Count',
 'device_id_Count',
 'platform_Count',
 'expedite',
 'fast',
 'quick' ]
target = 'target'

clf = RandomForestClassifier(n_jobs=2, random_state=0)

trained_model = clf.fit(train[training_features], train[target])

#print "Trained model :: ", trained_model
predictions = trained_model.predict(test[training_features])

headers = ["name", "score"]

values = sorted(zip(train[training_features].columns, trained_model.feature_importances_), key=lambda x: x[1] * -1)

print(values, headers)


################### Train and Test Accuracy


print "Train Accuracy :: ", accuracy_score(train[target], trained_model.predict(train[training_features]))
print "Test Accuracy  :: ", accuracy_score(test[target], trained_model.predict(test[training_features])


#######################  Holdout Sample ####################

hold_model = clf.fit(ht[training_features], ht[target])

print "Trained model :: ", hold_model
predictions = hold_model.predict(ht[training_features])

print "Hold Accuracy :: ", accuracy_score(ht[target], hold_model.predict(ht[training_features]))
 
################################ Precision  ###########


precision = cross_val_score(clf, train[training_features], train[target], scoring='precision')
recall = cross_val_score(clf, train[training_features], train[target], scoring='recall')

average_precision = average_precision_score(train[target], trained_model.predict(train[training_features]))
print('Average precision: {0:0.2f}'.format(
      precision))


