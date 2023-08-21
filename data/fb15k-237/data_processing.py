## Variables
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import geocoder
import argparse


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def get_continent(location, geonames_username):
    g = geocoder.geonames(location, key=geonames_username)
    loc = g.geonames_id
    g = geocoder.geonames(g.geonames_id, method='details', key=geonames_username)
    if g and g.ok:
        return g.continent

    return None
def main(args):
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
  relation = args.relation
  #relation = '/education/university/international_tuition./measurement_unit/dated_money_value/currency'
  #relation = '/education/university/domestic_tuition./measurement_unit/dated_money_value/currency' # no
  #relation = '/base/biblioness/bibs_location/state/continent'
  #relation = '/people/person/place_of_birth/continent'
  #relation = '/location/statistical_region/gdp_nominal_per_capita./measurement_unit/dated_money_value/currency'
  #relation = '/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency'
  #relation = '/education/university/local_tuition./measurement_unit/dated_money_value/currency'
  #relation = '/film/film/estimated_budget./measurement_unit/dated_money_value/currency'
  #relation = '/sports/sports_team/sport'
  #relation = '/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency'
  #relation = '/time/event/instance_of_recurring_event'
  #relation = '/film/film/estimated_budget./measurement_unit/dated_money_value/currency'
  #relation = '/location/administrative_division/first_level_division_of'
  #relation = '/location/administrative_division/country'
  #relation = '/organization/endowed_organization/endowment./measurement_unit/dated_money_value/currency'
  #t = /m/05bcl\t/location/administrative_division/first_level_division_of
  # code for labels not present in the dataset
  if relation == '/base/biblioness/bibs_location/country/continent' or relation == '/base/biblioness/bibs_location/state/continent' or relation == '/people/person/place_of_birth/continent':
    extract_relation = relation[:-10]
    # prendo solo le triple reltive alla relazione di interesse
    filtered_df = triplets[triplets['relation'] == extract_relation].drop('relation', axis=1).reset_index().drop('index', axis=1)
    # prendo tutte le possibili destinazioni
    possible_destinations_dict = {}
    possible_destinations = list(set(filtered_df['destination_entity'].to_list()))
    for i in range(0, len(possible_destinations)):
      possible_destinations_dict[possible_destinations[i]] = i
    # creo le labels
    continent_dictionary = {
      'EU': 0,
      'AS': 1,
      'SA': 2,
      'NA': 3,
      'AF': 4,
      'OC': 5,
      None: 6
    }
    continent_count = {
      'EU': 0,
      'AS': 0,
      'SA': 0,
      'NA': 0,
      'AF': 0,
      'OC': 0,
      None:0
    }
    labels = {}
    for index, row in filtered_df.iterrows():
      dest_name = df_entities_name.loc[df_entities_name['entity'] == row['destination_entity'], 'name'].values[0]
      if dest_name == 'United_States_of_America':
        continent = 'NA'
      elif dest_name == 'Republic_of_Macedonia' or dest_name == 'Palestinian_National_Authority':
        continent = 'AS'
      else:
        continent = get_continent(dest_name, 'frafrix97')
        print(continent, row['source_entity'])
      labels[row['source_entity']] = continent_dictionary[continent]
      continent_count[continent] +=1
    print('continent count ', continent_count)
    for k, v in continent_count.items():
      if v == 0:
          continent_dictionary.pop(k)
    if None in continent_dictionary:
      continent_dictionary.pop(None)
    relation = extract_relation
  else:
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
  print('temp df ', temp_df)


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
  print('new df ', new_df)
  # e lo concateno al dataframe di partenza
  triplets = pd.concat([triplets, new_df], ignore_index=True)

  # rimuovo dal dataframe delle entità della relazione
  entities = df_entities_name[~df_entities_name['entity'].isin(list(possible_destinations_dict.keys()))].reset_index().drop('index', axis=1)
  entities = entities.reset_index().drop('name', axis=1)

  # rimuovo le triple col la relazione
  new_triplets = triplets.drop(triplets[triplets['relation'] == relation].index).reset_index().drop('index', axis=1)
  new_new_triplets = new_triplets.drop(new_triplets[new_triplets['destination_entity'].isin(list(possible_destinations_dict.keys()))].index).reset_index().drop('index', axis=1)
  triplets = new_new_triplets.drop(new_new_triplets[new_new_triplets['source_entity'].isin(list(possible_destinations_dict.keys()))].index).reset_index().drop('index', axis=1)
  print(triplets)
  # creo il dizionario di features
  len_features = len(possible_destinations_dict.keys()) + 1
  embedding = np.zeros(len_features)
  #embedding = np.ones(10)/10

  new_entities = {valore: chiave for chiave, valore in entities['entity'].to_dict().items()}
  node_embeddings = {}
  for k, v in new_entities.items():
    #node_embeddings[k] = np.ones(10)/10
    node_embeddings[k] = embedding.copy()

  #for index, row in filtered_df.iterrows():
  #  node_embeddings[row['source_entity']][possible_destinations_dict[row['destination_entity']]] = 1.
  #  node_embeddings[row['source_entity']][-1] = 1

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
  temp_l = []
  for k, v in labels.items():
    if k not in new_entities and k not in temp_l:
      temp_l.append(k)
  print('temp_l: ', temp_l)
  #triplets = triplets[~triplets['source_entity'].isin(temp_l)]
  for k, v in labels.items():
    if k in new_entities: 
      new_labels[new_entities[k]] = v

  new_node_embeddings = {}
  for k, v in node_embeddings.items():
    new_node_embeddings[new_entities[k]] = v

  with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237/link.dat", "w+") as f:
    for index, row in triplets.iterrows():
        f.write(str(row['source_entity']) + '\t' + str(row['relation']) + '\t' + str(row['destination_entity']) + '\n')
  f.close()


  with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237/node.dat", "w+") as f:
    for index, value in new_node_embeddings.items():
        f.write(str(index) + '\t')
        for i in range(0, len(value.tolist())):
          f.write(str(value[i]) + '\t')
        f.write('\n')
  f.close()


  with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237/label.dat", "w+") as f:
    for index, value in new_labels.items():
        f.write(str(index) + '\t' + str(value) + '\n')
  f.close()

  with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237/relations_legend.dat", "w+") as f:
    for index, value in relation_dict.items():
        f.write(str(index) + '\t' + str(value) + '\n')
  f.close()

  ### ### ### ### ### ### ### ### ### ### 
  ### Text embedding
  df_text = pd.read_csv(descriptions, sep='\t', names=['entity', 'description'])
  df_text['node'] = ''

  for index, row in df_text.iterrows():
    if row['entity'] in new_entities:
      df_text.at[index, 'node'] = new_entities[row['entity']]
    else:
      df_text = df_text.drop(index)
  del df_text['entity']

  colonna_da_spostare = df_text.pop('node')

  # Inserisci la colonna come prima colonna nel DataFrame
  df_text.insert(0, 'node', colonna_da_spostare)
  descriptions_dict = dict(zip(df_text['node'], df_text['description']))

  def preprocess_text(text):
      # Rimozione della punteggiatura
      text = re.sub(r'[^\w\s]', '', text)

      # Conversione in minuscolo
      text = text.lower()

      # Rimozione delle stop words
      stop_words = set(stopwords.words('english'))
      tokens = word_tokenize(text)
      filtered_tokens = [word for word in tokens if word not in stop_words]
      text = ' '.join(filtered_tokens)

      # Lemmatization
      lemmatizer = WordNetLemmatizer()
      tokens = word_tokenize(text)
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
      text = ' '.join(lemmatized_tokens)

      return text

  def preprocess_dataset(dataset):
      preprocessed_dataset = {}
      for entity, description in dataset.items():
          preprocessed_description = preprocess_text(description)
          preprocessed_dataset[entity] = preprocessed_description
      return preprocessed_dataset

  preprocessed_dataset = preprocess_dataset(descriptions_dict)

  ## Bags of Words embedding
  def bow_embedding(dataset, num_components=100):
      vectorizer = CountVectorizer()
      bow_features = vectorizer.fit_transform(dataset.values())

      # Applica PCA per ridurre la dimensione dell'embedding
      pca = PCA(n_components=num_components)
      bow_features_pca = pca.fit_transform(bow_features.toarray())

      embedding_dict = {}
      for entity, embedding in zip(dataset.keys(), bow_features_pca):
          embedding_dict[entity] = embedding

      return embedding_dict
  # BoW embedding
  bow_embeddings = bow_embedding(preprocessed_dataset)

  with open("/Users/francescoferrini/VScode/MultirelationalGNN/data/fb15k-237/node_bow.dat", "w") as f:
    for index, value in new_node_embeddings.items():
        f.write(str(index) + '\t')
        if index in bow_embeddings:
          for i in range(0, len(bow_embeddings[index].tolist())):
            f.write(str(bow_embeddings[index][i]) + '\t')
          f.write('\n')
        else:
          fe = np.zeros(100)
          for i in range(0, len(fe.tolist())):
            f.write(str(fe[i]) + '\t')
          f.write('\n')
  f.close()

  print('end')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fbk15k-237 graph creation')
    parser.add_argument("--relation", type=str, required=True,
            help="relation transformed into label")
    


    args = parser.parse_args()
    #print(args, flush=True)
    main(args)