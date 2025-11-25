import pandas as pd
import numpy as np
import seaborn as sns

env = pd.read_csv('./data/environmental_activities.csv')
rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
sustain = pd.read_csv('./data/sustainable_development_goals.csv')
train = pd.read_csv('./data/train.csv')

# with scope per usd

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

vlag_cmap = sns.color_palette("vlag", as_cmap=True)

fig = sns.heatmap(train_coeff,
                  cbar=False,
                  square=True,
                  xticklabels=False,
                  yticklabels=True,
                  annot=False,
                  cmap=vlag_cmap,
                  center=0).get_figure()
fig.tight_layout()
fig.savefig('./feature/heatmap.png')
fig.show()

# with just scope

train2 = pd.merge(train,
                  rev.drop(['nace_level_1_name',
                           'nace_level_2_code',
                           'nace_level_2_name'], axis=1),
                           how='inner', on='entity_id')

train2['revenue_pct_new'] = train2.groupby(['entity_id', 'nace_level_1_code'])['revenue_pct'].transform('sum')
train2 = train2.drop(['revenue_pct'], axis=1).drop_duplicates(keep='first')

train2 = train2[['entity_id',
               'environmental_score',
               'social_score',
               'governance_score',
               'nace_level_1_code',
               'revenue_pct_new',
               'target_scope_1',
               'target_scope_2']]

train2_pivot = train2.pivot_table(
    index='entity_id',
    columns='nace_level_1_code', 
    values='revenue_pct_new', 
    aggfunc='first'
)

train2 = pd.merge(train2, train2_pivot, how = 'inner', on = 'entity_id').drop(['revenue_pct_new', 'entity_id', 'nace_level_1_code'], axis=1).drop_duplicates(keep='first').fillna(value=0)

train2_t = train2.transpose()

# stops working

train2_coeff = np.corrcoef(train2_t)
train2_coeff = pd.DataFrame(train2_coeff, index=train2.columns, columns=train2.columns)

train2_coeff.to_csv('./feature/train2_coeff.csv')

vlag_cmap = sns.color_palette("vlag", as_cmap=True)

fig2 = sns.heatmap(train2_coeff,
                  cbar=False,
                  square=True,
                  xticklabels=False,
                  yticklabels=True,
                  annot=False,
                  cmap=vlag_cmap,
                  center=0).get_figure()
fig2.tight_layout()
fig2.savefig('./feature/heatmap2.png')
fig2.show()