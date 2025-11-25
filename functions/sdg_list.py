import pandas as pd

test_file = pd.read_csv("data/test.csv")

sdg_file = pd.read_csv("data/sustainable_development_goals.csv")

print("***********\n List of Entities and the goals they have publicly committed to:\n")
# # print(sdg_file.head())
# # print(sdg_file)

# # entity_ids as a set
# entity_ids = set(sdg_file['entity_id'])
# # print(entity_ids)
# # print(len(entity_ids)) .... 130 unique entity_ids

# # entity_ids as a list ** for traversal
# entity_ids_list = list(entity_ids)
# # print(entity_ids_list)

# # sdg_ids as a set
# sdg_ids = set(sdg_file['sdg_id'])
# print(sdg_ids)

# #sdg_ids as a list ** for traversal
# sdg_ids_list = list(sdg_ids)



# Dictionary : [entity_id] = {sdg_set}

entity_dict = sdg_file.groupby('entity_id')['sdg_id'].apply(set).to_dict()
for entity_id, sdg_set in list(entity_dict.items()):
    print(f"{entity_id} : {sdg_set}")
print(f"\nNumber of Entities: {len(entity_dict)}")

test_id = set(test_file['entity_id'])
sdg_id = set(sdg_file['entity_id'])

common_entities = test_id.intersection(sdg_id)

print(f"Common Entities in test and sdg file: {common_entities}")
print(f"Number of Common Entities {len(common_entities)}")


print("\n\nSDG Scores:")
# SDG Score : [0,3]

sdg_scores = {} # dictionary

# entity_dict:
# [entity_id] = {sdg_set}
# for entity_id, sdg_set in entity_dict.items():
#     if entity_id in common_entities:
#         num_sdgs = len(sdg_set)
#         sdg_scores[entity_id] = num_sdgs

#     else: 
#         sdg_scores[entity_id] = 0

#     for entity_id, score in sdg_scores.items():
#         if score == 0:
#             print(f"Entity {entity_id}: Score = 0")
#         else:
#             print(f"Entity {entity_id}: Score = {score}")

