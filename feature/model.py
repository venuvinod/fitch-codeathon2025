# setup

import pandas as pd

env = pd.read_csv('./data/environmental_activities.csv')
rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
sustain = pd.read_csv('./data/sustainable_development_goals.csv')
train = pd.read_csv('./data/train.csv')

# emissions per dollar revenue

train['scope_1_per_usd'] = train['target_scope_1'] / train['revenue']
train['scope_2_per_usd'] = train['target_scope_2'] / train['revenue']

# emissions per dollar revenue for each industry

# adds nace_level_1_code and revenue_pct to train
train = pd.merge(train,
                 rev[['entity_id', 'nace_level_1_code', 'revenue_pct']],
                 how='inner', on='entity_id')

# sums revenue_pct for each group (entity_id, nace_level_1_code)
train['revenue_pct_new'] = train.groupby(['entity_id', 'nace_level_1_code'])['revenue_pct'].transform('sum')
train = train.drop(['revenue_pct'], axis=1).drop_duplicates(keep='first')

# calculates scope per usd revenue weighted by revenue % for each nace
train['scope_1_per_usd_nace'] = train['scope_1_per_usd'] * train['revenue_pct_new']
train['scope_2_per_usd_nace'] = train['scope_2_per_usd'] * train['revenue_pct_new']

# creates df of mean emissions per usd revenue by nace across all entities
emissions_per_usd_nace = pd.DataFrame()
emissions_per_usd_nace['scope_1'] = train.groupby('nace_level_1_code')['scope_1_per_usd_nace'].agg('mean')
emissions_per_usd_nace['scope_2'] = train.groupby('nace_level_1_code')['scope_2_per_usd_nace'].agg('mean')

# calculating accuracy of estimate on train

estimate_train = pd.read_csv('./data/train.csv').drop(['target_scope_1', 'target_scope_2'], axis=1)

estimate_train = pd.merge(estimate_train,
                 rev[['entity_id', 'nace_level_1_code', 'revenue_pct']],
                 how='inner', on='entity_id')