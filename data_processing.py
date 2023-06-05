## Variables
import pandas as pd
import numpy as np

path = '/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237'
entities_name = path + '/FB15k_mid2name.txt'
train_triplets = path + '/train.tsv'
test_triplets = path + '/test.tsv'
validation_triplets = path + '/dev.tsv'
descriptions = path + '/entity2textlong.txt'

df_entities_name = pd.read_csv(entities_name, sep='\t', names=['entity', 'name'])
df_train_triplets = pd.read_csv(train_triplets, sep='\t', names=['source_entity', 'relation', 'destination_entity'])
df_test_triplets = pd.read_csv(test_triplets, sep='\t', names=['source_entity', 'relation', 'destination_entity'])
df_val_triplets = pd.read_csv(validation_triplets, sep='\t', names=['source_entity', 'relation', 'destination_entity'])
df_text = pd.read_csv(descriptions, sep='\t', names=['entity', 'description'])

# put together all the triplets in the 3 files
frames = [df_train_triplets, df_test_triplets, df_val_triplets]
triplets = pd.concat(frames, axis=0, ignore_index=True)

## Search for many-to-one relations
# Raggruppamento dei source_entity per relation
grouped = triplets.groupby('relation')['source_entity'].agg(list).reset_index()
# Rimozione delle righe con liste contenenti doppioni
grouped_n = grouped[grouped['source_entity'].apply(lambda x: len(x) == len(set(x)))]
relation_list = grouped_n['relation'].tolist()
print(relation_list)

## Create the new dataset
# specifico la relazione che diventa label/attributo.    /music/group_member/membership./music/group_membership/group
relation = '/people/person/gender'

# dal dataframe di triple prendo solo quelle relative a quella relazione
filtered_df = triplets[triplets['relation'] == relation].drop('relation', axis=1).reset_index().drop('index', axis=1)

# prendo tutte le possibili entità che quella relazione puo assumere
possible_destinations_dict = {}
possible_destinations = list(set(filtered_df['destination_entity'].to_list()))
for i in range(0, len(possible_destinations)):
  possible_destinations_dict[possible_destinations[i]] = i
print(possible_destinations_dict)

# e creo un dizionario di labels
labels = {}
for index, row in filtered_df.iterrows():
  labels[row['source_entity']] = possible_destinations_dict[row['destination_entity']]

# check how those classes are populated
# Filtraggio del dataframe per la relation di interesse
temp_df = triplets[triplets['relation'] == relation]

# Calcolo del conteggio per ogni elemento nella lista di destination_entity
count_dict = {}
for destination in list(possible_destinations_dict.keys()):
    count = temp_df[temp_df['destination_entity'] == destination].shape[0]
    count_dict[destination] = count

# Stampa dei risultati
for destination, count in count_dict.items():
    print(f"La relation '{relation}' compare {count} volte con destination_entity '{destination}'")

# creo un nuovo dataframe per legare i nodi slegati
new_df = pd.DataFrame(columns=['source_entity', 'relation', 'destination_entity'])
source_entities = list(set(filtered_df['source_entity']))
data = []
for index, row in triplets.iterrows():
  if row['source_entity'] in possible_destinations:
    for elm in source_entities:
      data.append([elm, row['relation'], row['destination_entity']])
new_df = pd.DataFrame(data, columns=['source_entity', 'relation', 'destination_entity'])

# e lo concateno al dataframe di partenza
triplets = pd.concat([triplets, new_df], ignore_index=True)

# rimuovo dal dataframe delle entità le entita della relazione
entities = df_entities_name[~df_entities_name['entity'].isin(list(possible_destinations_dict.keys()))].reset_index().drop('index', axis=1)
entities = entities.reset_index().drop('name', axis=1)

# rimuovo le triple col la relazione
new_triplets = triplets.drop(triplets[triplets['relation'] == relation].index).reset_index().drop('index', axis=1)
new_new_triplets = new_triplets.drop(new_triplets[new_triplets['destination_entity'].isin(list(possible_destinations_dict.keys()))].index).reset_index().drop('index', axis=1)
triplets = new_new_triplets.drop(new_new_triplets[new_new_triplets['source_entity'].isin(list(possible_destinations_dict.keys()))].index).reset_index().drop('index', axis=1)

# creo il dizionario di features
len_features = len(possible_destinations_dict.keys()) + 1
embedding = np.zeros(len_features)
#embedding = np.ones(10)/10

new_entities = {valore: chiave for chiave, valore in entities['entity'].to_dict().items()}
node_embeddings = {}
for k, v in new_entities.items():
  #node_embeddings[k] = np.ones(10)/10
  node_embeddings[k] = embedding.copy()

for index, row in filtered_df.iterrows():
  node_embeddings[row['source_entity']][possible_destinations_dict[row['destination_entity']]] = 1.
  node_embeddings[row['source_entity']][-1] = 1

# relations
relation_list = list(set(triplets['relation'].to_list()))
#relation_list.remove(relation)
relation_dict = {}
for i in range(0, len(relation_list)):
  relation_dict[relation_list[i]] = i

# triplets
for index, row in triplets.iterrows():
  triplets.at[index, 'source_entity'] = new_entities[row['source_entity']]
  triplets.at[index, 'relation'] = relation_dict[row['relation']]
  triplets.at[index, 'destination_entity'] = new_entities[row['destination_entity']]

## Create files
new_labels = {}
for k, v in labels.items():
  new_labels[new_entities[k]] = v

new_node_embeddings = {}
for k, v in node_embeddings.items():
  new_node_embeddings[new_entities[k]] = v

with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k237/link.dat", "w") as f:
  for index, row in triplets.iterrows():
      f.write(str(row['source_entity']) + '\t' + str(row['relation']) + '\t' + str(row['destination_entity']) + '\n')
f.close()

with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k237/node.dat", "w") as f:
  for index, value in new_node_embeddings.items():
      f.write(str(index) + '\t')
      for i in range(0, len(value.tolist())):
        f.write(str(value[i]) + '\t')
      f.write('\n')
f.close()

with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k237/label.dat", "w") as f:
  for index, value in new_labels.items():
      f.write(str(index) + '\t' + str(value) + '\n')
f.close()

print(relation_dict)