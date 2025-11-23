import pandas as pd
import numpy as np
import seaborn as sns

env = pd.read_csv('./data/environmental_activities.csv')
rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
sustain = pd.read_csv('./data/sustainable_development_goals.csv')
train = pd.read_csv('./data/train.csv')

train = pd.merge(train,
                 rev.drop(['nace_level_1_name',
                           'nace_level_2_code',
                           'nace_level_2_name'], axis=1),
                how='inner', on='entity_id')

train['revenue_pct_new'] = train.groupby(['entity_id', 'nace_level_1_code'])['revenue_pct'].transform('sum')
train = train.drop(['revenue_pct'], axis=1).drop_duplicates(keep='first')

train['scope1_per_usd'] = train['target_scope_1'] / train['revenue'] * 100000
train['scope2_per_usd'] = train['target_scope_2'] / train['revenue'] * 100000
train = train[['entity_id',
               'environmental_score',
               'social_score',
               'governance_score',
               'nace_level_1_code',
               'revenue_pct_new',
               'scope1_per_usd',
               'scope2_per_usd']]

train_pivot = train.pivot_table(
    index='entity_id',
    columns='nace_level_1_code', 
    values='revenue_pct_new', 
    aggfunc='first'
)

train = pd.merge(train, train_pivot, how = 'inner', on = 'entity_id').drop(['revenue_pct_new', 'entity_id', 'nace_level_1_code'], axis=1).drop_duplicates(keep='first').fillna(value=0)

train_t = train.transpose()

train_coeff = np.corrcoef(train_t)
train_coeff = pd.DataFrame(train_coeff, index=train.columns, columns=train.columns)

train_coeff.to_csv('./feature/train_coeff.csv')

fig = sns.heatmap(train_coeff, cbar=True, square=True, xticklabels=False, yticklabels=True, annot=False).get_figure()
fig.tight_layout()
fig.savefig('./feature/heatmap.png')
fig.show()