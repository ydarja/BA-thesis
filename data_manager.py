import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import random
import csv
import time
import pandas as pd
from sklearn.model_selection import train_test_split
def get_depth(synset):
    # Get the maximum depth for the synset
    max_depth = max(len(path) for path in synset.hypernym_paths())
    return max_depth

def sample_data(num_pairs):

    random.seed(42)
    data = []

    while len(data) < num_pairs:
        synset1 = random.choice(list(wn.all_synsets()))

        # get depth of synset1
        depth1 = len(synset1.hypernym_paths())

        # get example sentence of synset1
        examples1 = synset1.examples()
        if not examples1:
            continue
        sentence1 = examples1[0]

        # choose a random hypernym from synset1
        hypernyms = synset1.hypernyms()
        if not hypernyms:
            continue
        hypernym1 = random.choice(hypernyms)

        # get synsets that are deeper in the hierarchy than synset1 by comparing to common hypernym
        deeper_synsets = [s for s in wn.all_synsets() if s != synset1 and s != hypernym1 and s.shortest_path_distance(hypernym1) is not None and s.shortest_path_distance(hypernym1) > 0 and s.shortest_path_distance(s) is not None and s.shortest_path_distance(s) > s.shortest_path_distance(synset1)]
        if not deeper_synsets:
            continue

        synset2 = random.choice(deeper_synsets)

        # get depth of synset2
        depth2 = len(synset2.hypernym_paths())

        # get example sentence of synset2
        examples2 = synset2.examples()
        if not examples2:
            continue
        sentence2 = examples2[0]

        data.append({
            'id': len(data) + 1,
            'synset1': synset1,
            'depth1': depth1,
            'sentence1': sentence1,
            'synset2': synset2,
            'depth2': depth2,
            'sentence2': sentence2
        })

    return data


def filter_synsets():
    # Filter out synsets without examples and not nouns (done once)
    return [synset for synset in list(wn.all_synsets('n')) if synset.examples()]


def simple_sample_data(num_pairs, filtered_synsets):
    random.seed(42)
    data = []
    start_time = time.time()

    while len(data) < num_pairs:


        # pick two random synsets
        synset1 = random.choice(list(filtered_synsets))
        synset2 = random.choice(list(filtered_synsets))

        # get example sentences of synset1 and synset2
        examples1 = synset1.examples()
        examples2 = synset2.examples()

        # get depths of the two synsets
        depth1 = len(synset1.hypernym_paths())
        depth2 = len(synset2.hypernym_paths())

        # skip synstes on the same depth
        if depth1 == depth2:
            continue

        #  which synset is more specific (deeper)
        specific = 0 if depth1 > depth2 else 1
 
        data.append({
            'id': len(data) + 1,
            'synset1': synset1,
            'depth1': depth1,
            'sentence1': examples1[0],
            'synset2': synset2,
            'depth2': depth2,
            'sentence2': examples2[0],
            'specific': specific
        })

        last_id = data[-1]["id"]
        if  last_id % 1000 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_instance = elapsed_time / last_id
                remaining_time = avg_time_per_instance * (num_pairs - last_id)
                print("Estimated remaining time: ", remaining_time)

    return data    

def save_data_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'synset1', 'depth1', 'sentence1', 'synset2', 'depth2', 'sentence2', 'specific'])
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)

def load_data_from_csv(file_path, test_size=0.2, val_size=0.1):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Further split the training data into training and validation sets
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    
    return train_data, val_data, test_data

if __name__ == "__main__":

    num_pairs = 8000
    #synsets = filter_synsets()
    # data = load_data_from_csv('data/synset_data.csv')

    df = pd.read_csv('data/synset_data.csv')
    s1 = list(df['sentence1'])
    s2 = list(df['sentence2'])
    s = s1+s2
    total_length = sum(len(sentence) for sentence in s)
    average_length = total_length / len(s)
    print('Average sentence length:', average_length)
    """
    # Calculate the average sentence length
    total_length = 0
    sentence_count = 0

    for _, entry in data.iterrows():
        total_length += len(entry['sentence1'].split()) + len(entry['sentence2'].split())
        sentence_count += 2

    average_length = total_length / sentence_count
    print('Average sentence length:', average_length)
                
    """  
