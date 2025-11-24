import pandas as pd
import numpy as np  


train = pd.read_csv("./data/train.csv")
train_modified = pd.read_csv('./feature/train_modified.csv')


data_per_cc = (train_modified.groupby('country_code').agg(
    num_entities=('entity_id', 'nunique'),
    avg_scope1_per_usd = ('scope1_per_usd', 'mean'),
    avg_scope2_per_usd = ('scope2_per_usd','mean'))).reset_index()

data_per_cc = data_per_cc.sort_values('num_entities', ascending = False)

data_per_cc.to_csv('./feature/emissions_avg.csv')
