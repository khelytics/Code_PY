
### Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import json

#### Load pandas
import pandas as pd

#### Load numpy
import numpy as np

#### Set random seed
np.random.seed(0)

##########################################################################
################### Import  Data  #################################
###############################################################################

########################### Import Event Data  ######################

d1 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q12017/part-r-00000-b56e2d37-023a-4935-803e-9ccb43f25d3b.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'},nrows=25)
d2 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q22017/part-r-00000-00fb46cf-f978-4ac4-a171-8acb3d8c38df.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'},nrows=25)
d3 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q32017/part-r-00000-e840147f-6ae5-4970-8db6-e874ff68aed6.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'},nrows=25)
d4 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q42017/NEWpart-r-00000-e3d4e4e6-8ce8-4782-a8c2-6f0727b9dade.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'},nrows=25)

d_hold = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q42017/holdOutSampleHeader.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'},nrows=25)



### Append all rows in to one dataset

data=pd.concat([d1,d2,d3,d4])


##### Create binary variables for all evt_type_cd

data_dummy = pd.concat([data, pd.get_dummies(data['evt_type_cd'])], axis=1);
d_hold_dummy = pd.concat([d_hold, pd.get_dummies(d_hold['evt_type_cd'])], axis=1);

#### Pick the correct format of Timestamp columns
data_dummy['event_ts']=pd.to_datetime(data_dummy['evt_ts'])
data_dummy['monet_ts']=pd.to_datetime(data_dummy['monetization_ts'])

d_hold_dummy['event_ts']=pd.to_datetime(d_hold_dummy['evt_ts'])
d_hold_dummy['monet_ts']=pd.to_datetime(d_hold_dummy['monetization_ts'])


##### Select data columns for modeling 

mod_data = data_dummy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '043',
 '173',
 '176',
 '191',
 '466',
 '991',
 '992',
 '994'  ]].copy()
 
 d_hold_dummy
 hol_data = d_hold_dummy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466' ]].copy()
 
 
######### Import Device Data

dev = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/Device/part-r-00000-52c9d811-dd02-45f4-9f0e-d67c18e339ba.csv",dtype={'acct_id':'object','device_id':'object','platform':'object','first_login':'object','last_login':'object'}, nrows=25)

#### Pick two columns : Account_ID and device ID

dev_n = dev.iloc[:,0:2] 

#### Count the account per device ID

device_count=dev_n.groupby("device_id").agg({"Acct_count": lambda x: x.nunique()})

pd.DataFrame(dev_n.groupby("device_id").agg({"Acct_count": lambda x: x.nunique()}).reset_index()
pd.merge(dev_n, device_count, on='subject_id', how='left') 
 
 
 
### Replace NaN with Zero
mod_data['timedif'].fillna(0, inplace=True)



################ Create Training And Test Data(70% Train & 30% Test)


mod_data['is_train'] = np.random.uniform(0, 1, len(mod_data)) <= .70

# Create two new dataframes, one with the training rows, one with the test rows

train, test = mod_data[mod_data['is_train']==True], mod_data[mod_data['is_train']==False]

############## Show the number of observations for the test and training dataframes

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print('Number of observations in the holdout data:',len(hol_data))


###### Select feature and target Variable

training_features = ['012', 
 '013',
 '020',
 '024',
 '042',
 '043',
 '173',
 '176',
 '191',
 '466',
 '991',
 '992',
 '994']
target = 'target'


training_features = ['012', 
 '013',
 '020',
 '024',
 '042',
 '043',
 '173',
 '176',
 '191',
 '466',
 '991',
 '992',
 '994']
target = 'target'
 
########## Random Forest Approach in  scikit-learn

## Train Model

clf = RandomForestClassifier(n_jobs=2, random_state=0)

trained_model = clf.fit(train[training_features], train[target])

print "Trained model :: ", trained_model
predictions = trained_model.predict(test[training_features])


#### Which feature is more important : High the value, better predictor it is.

headers = ["name", "score"]

values = sorted(zip(train[training_features].columns, trained_model.feature_importances_), key=lambda x: x[1] * -1)

print(tabulate(values, headers, tablefmt="plain"))


################### Train and Test Accuracy

print "Train Accuracy :: ", accuracy_score(train[target], trained_model.predict(train[training_features]))
print "Test Accuracy  :: ", accuracy_score(test[target], predictions)




