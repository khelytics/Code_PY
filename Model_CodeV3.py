
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

##### read ev Data

d1 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q12017/part-r-00000-b56e2d37-023a-4935-803e-9ccb43f25d3b.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d2 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q22017/part-r-00000-00fb46cf-f978-4ac4-a171-8acb3d8c38df.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d3 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q32017/part-r-00000-e840147f-6ae5-4970-8db6-e874ff68aed6.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})
d4 = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q42017/NEWpart-r-00000-e3d4e4e6-8ce8-4782-a8c2-6f0727b9dade.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})

d_hold = pd.read_csv("G:/Upwork/Hackathon/ML-Hackathon/ATO TSYS Event & IVR/Q42017/holdOutSampleHeader.csv",dtype={'acct_id':'object','evt_type_cd':'object','evt_ts':'object','monetization_ts':'object','target':'object'})

data=pd.concat([d1,d2,d3,d4])
data_dummy = pd.concat([data, pd.get_dummies(data['evt_type_cd'])], axis=1);
d_hold_dummy = pd.concat([d_hold, pd.get_dummies(d_hold['evt_type_cd'])], axis=1);

#### Pick the correct format of Timestamp columns
data_dummy['event_ts']=pd.to_datetime(data_dummy['evt_ts'])
data_dummy['monet_ts']=pd.to_datetime(data_dummy['monetization_ts'])

d_hold_dummy['event_ts']=pd.to_datetime(d_hold_dummy['evt_ts'])
d_hold_dummy['monet_ts']=pd.to_datetime(d_hold_dummy['monetization_ts'])

data_dummy['freq'] = data_dummy.groupby('acct_id')['acct_id'].transform('count')
d_hold_dummy['freq'] = d_hold_dummy.groupby('acct_id')['acct_id'].transform('count')
mod_data = data_dummy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'freq' ]].copy()
 
hol_data = d_hold_dummy[[
 'target',
 '012',
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
 'freq' ]].copy()
mod_data['is_train'] = np.random.uniform(0, 1, len(mod_data)) <= .70

train, test = mod_data[mod_data['is_train']==True], mod_data[mod_data['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print('Number of observations in the holdout data:',len(hol_data))

training_features = ['012', 
 '013',
 '020',
 '024',
 '042',
 '173',
 '191',
 '466',
'freq'  ]
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

