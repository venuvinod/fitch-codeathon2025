import pandas as pd
import numpy as np

env = pd.read_csv('./data/environmental_activities.csv')
rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
sustain = pd.read_csv('./data/sustainable_development_goals.csv')
test = pd.read_csv('./data/test.csv')

# find intersect of test ids with env/rev/sustain

env_id = set(env['entity_id'])
rev_id = set(rev['entity_id'])
sustain_id = set(sustain['entity_id'])
test_id = set(test['entity_id'])

intersect = test_id.intersection(env_id, rev_id, sustain_id)
print(intersect)
print(len(intersect)) # 7 

intersect_env = test_id.intersection(env_id)
print(intersect_env)
print(len(intersect_env)) # 23

intersect_rev = test_id.intersection(rev_id)
print(intersect_rev)
print(len(intersect_rev)) # 49

intersect_sustain = test_id.intersection(sustain_id)
print(intersect_sustain)
print(len(intersect_sustain)) # 12

# quantify revenue per sector in rev
rev_sector = pd.merge(rev, test, how='inner', on='entity_id')[['entity_id',
                                                               'nace_level_1_code',
                                                               'nace_level_2_code',
                                                               'revenue_pct',
                                                               'revenue']]
rev_sector['revenue_per_sector'] = rev_sector['revenue_pct'] * rev_sector['revenue']
rev_sector = rev_sector.drop(['revenue', 'revenue_pct'], axis=1)

rev_sector.to_csv('./feature/rev_per_sector.csv')

# verify that the sum of revenue in rev_sector matches the revenue in test

# add revenue from test to rev