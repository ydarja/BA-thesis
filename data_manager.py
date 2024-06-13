import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import random
import csv
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_depth(synset):
    # get the maximum depth for the synset
    max_depth = max(len(path) for path in synset.hypernym_paths())
    return max_depth

def get_descendants(synset, visited=None):
    if visited is None:
        visited = set()

    descendants = set()
    
    visited.add(synset)  

    for hyponym in synset.hyponyms():
        if hyponym not in visited:  
            descendants.add(hyponym)
            descendants |= get_descendants(hyponym, visited)

    return descendants

def sample_data(num_pairs, filtered_synsets, id):
    random.seed(42)
    data = []
    start_time = time.time()

    while len(data) < num_pairs:
        synset1 = random.choice(filtered_synsets)

        # Check if synset1 is a leaf node
        if not synset1.hyponyms():
            continue  # Skip leaf nodes

        descendants = get_descendants(synset1)

        if not descendants:
            continue

        synset2 = random.choice(list(descendants))

        # get depth of synset1 and synset2
        depth1 = get_depth(synset1)
        depth2 = get_depth(synset2)

        # get example sentence of synset1 and synset2
        examples1 = synset1.examples()
        examples2 = synset2.examples()

        if not examples1 or not examples2:
            continue

        sentence1 = examples1[0] 
        sentence2 = examples2[0]

        specific = 0 if depth1 > depth2 else 1 

        data.append({
            'id': id + 1,
            'synset1': synset1,
            'depth1': depth1,
            'sentence1': sentence1,
            'synset2': synset2,
            'depth2': depth2,
            'sentence2': sentence2,
            'specific': specific
        })
        
        id += 1

        last_id = data[-1]["id"]
        if last_id % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_instance = elapsed_time / last_id
            remaining_time = avg_time_per_instance * (num_pairs - last_id)
            print("Estimated remaining time: ", remaining_time)

    return data


def filter_synsets_pos(pos, synsets):
    # select synsets of specific pos tag
    return [synset for synset in synsets if synset.pos() == pos]

def filter_synsets_examples():
    # filter out synsets without examples and not nouns (done once)
    return [synset for synset in list(wn.all_synsets()) if synset.examples()]
    

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

def balance_data(file):

    # Load the data
    data = pd.read_csv(file)

    
    num_rows = len(data)
    swap = num_rows // 2

    # Randomly select indices for the rows to be swapped
    np.random.seed(42)  # For reproducibility
    indices_to_swap = np.random.choice(num_rows, swap, replace=False)

    # Swap the columns for the selected rows
    data.loc[indices_to_swap, ['synset1', 'synset2']] = data.loc[indices_to_swap, ['synset2', 'synset1']].values
    data.loc[indices_to_swap, ['depth1', 'depth2']] = data.loc[indices_to_swap, ['depth2', 'depth1']].values
    data.loc[indices_to_swap, ['sentence1', 'sentence2']] = data.loc[indices_to_swap, ['sentence2', 'sentence1']].values

    # Update the specific label for the selected rows
    data.loc[indices_to_swap, 'specific'] = 0

    # Save the balanced DataFrame back to a CSV file
    balanced_file_path = 'data/balanced_raw_synsets.csv'
    data.to_csv(balanced_file_path, index=False)


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

    # Adjectives and adverbs don't have hypornym relations, so we don't include them
    '''
    n 8742
    v 9691
    a 4355
    r 3192
    '''
    '''
    num_pairs_per_pos = {
    'n': 8000,  
    'v': 9000}

    filtered = filter_synsets_examples()
    print("Filtered synsets without examples")
    data = []  
    for pos, num_pairs in num_pairs_per_pos.items():
        filtered_synsets = filter_synsets_pos(pos, filtered)
        print(pos, len(filtered_synsets))
        print("Processing: ", pos)
        data += sample_data(num_pairs, filtered_synsets, id=len(data))


    save_data_to_csv(data, 'raw_synsets.csv')
    '''
    balance_data('data/raw_synsets.csv')
    

    
