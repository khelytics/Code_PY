
import json
import pandas as pd

###### Load ATO Speak Easy Data
se_ato = []  # a list of all objects
with open('/home/ec2-user/hackathon/SpeakEasy/ato_speakeasy_transcripts.json') as ato_data:
  for line in ato_data:
    se_ato.append( json.loads(line)) 
	
ato_df=pd.DataFrame(se_ato)	

ato_df.columns.values

"""array(['call_type', 'calliope_additional_products', 'calliope_ae_areas',
       'calliope_card_terms', 'calliope_customer',
       'calliope_fraud_management', 'calliope_inbound_payments',
       'calliope_line_management', 'calliope_other', 'calliope_payments',
       'calliope_product_offer', 'calliope_rewards', 'complete_call_id',
       'duration', 'hub', 'icm_contact_id', 'id', 'is_acct_ato',
       'model_version', 'segment_count', 'segment_id', 'segments_in_call',
       'speakers', 'start_time', 'tag', 'transcript'], dtype=object)
	   """
	   
###### Load NON-ATO Speak Easy Data
se_nato = []  # a list of all objects

with open('/home/ec2-user/hackathon/SpeakEasy/nonato_speakeasy_transcripts.json') as nato_data:
  for line in nato_data:
    se_nato.append( json.loads(line) ) 

nato_df=pd.DataFrame(se_nato)
	

########################################################################################################
###################### Import Mapping for ATO SE And Assign Account IDs #######################
#######################################################################################################
import pandas as pd
ato_map = pd.read_csv("/home/ec2-user/hackathon/SpeakEasy/ATO_icm_contact_id_TO_acct_id_mapping.csv")	
ato_map
ato_map.columns.values
array(['CNTCT_ID', 'SEG_ACCT_ID'], dtype=object)
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
ato_df['icm_contact_id']

#### Count of Data  and mapping file
#ato_map[ato_map['icm_contact_id']==1522440000333974]
#ato_df[ato_df['icm_contact_id']==1522440000333974]
len(ato_map['icm_contact_id'].value_counts())
len(ato_df['icm_contact_id'].value_counts())

###### Asssign Account IDs to the data(Merging)
ato_mapped_data= pd.merge(ato_df, ato_map, on=['icm_contact_id'])

########################################################################################################
###################### Import Mapping for ATO SE And Assign Account IDs #######################
#######################################################################################################
import pandas as pd
nato_map = pd.read_csv("/home/ec2-user/hackathon/SpeakEasy/NonATO_icm_contact_id_TO_acct_id_mapping.csv")	
nato_map
nato_map.columns.values
#array(['CNTCT_ID', 'SEG_ACCT_ID'], dtype=object)
### Rename the Contact ID
nato_map.rename(columns={'CNTCT_ID':'icm_contact_id'}, inplace=True)
#len(ato_map)
#892569
### Remove Duplicates
nato_map = nato_map.drop_duplicates(subset=['icm_contact_id','SEG_ACCT_ID'])
 #len(ato_map)
#576742
#### Change the Column data Type 
#ato_map['icm_contact_id'].apply(str)
nato_df['icm_contact_id']=nato_df['icm_contact_id'].astype(int)
nato_df['icm_contact_id']

#### Count of Data  and mapping file
#ato_map[ato_map['icm_contact_id']==1521920000055214]
#nato_df[nato_df['icm_contact_id']==1521890000412478]
len(nato_map['icm_contact_id'].value_counts())
len(nato_df['icm_contact_id'].value_counts())

###### Asssign Account IDs to the data(Merging)
nato_mapped_data= pd.merge(nato_df, nato_map, on=['icm_contact_id'])

nato_data
se_ato_nato=pd.concat([nato_mapped_data,ato_mapped_data])



