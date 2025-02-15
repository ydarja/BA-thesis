import nltk
from nltk.corpus import wordnet as wn
import random
import csv
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

############### RAW DATA ####################

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

        # skip leaf nodes
        if not synset1.hyponyms():
            continue  

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

def balance_data(file, balanced_file):
    data = pd.read_csv(file)

    num_rows = len(data)
    swap = num_rows // 2

    # randomly select indices for the rows to be swapped
    np.random.seed(42) 
    indices_to_swap = np.random.choice(num_rows, swap, replace=False)

    # swap the columns for the selected rows
    data.loc[indices_to_swap, ['synset1', 'synset2']] = data.loc[indices_to_swap, ['synset2', 'synset1']].values
    data.loc[indices_to_swap, ['depth1', 'depth2']] = data.loc[indices_to_swap, ['depth2', 'depth1']].values
    data.loc[indices_to_swap, ['sentence1', 'sentence2']] = data.loc[indices_to_swap, ['sentence2', 'sentence1']].values
    data.loc[indices_to_swap, 'specific'] = 0

    data.to_csv(balanced_file)

    return data


############## SAME CONTEXT DATA ###############

def extract_target_word(synset):
    """Extract target word from synset string"""
    synset_pattern = r"Synset\('([^\.]+)\."
    match = re.search(synset_pattern, synset)
    if match:
        base_word = match.group(1).replace('_', ' ').lower()
        return base_word
    return None

def lemmatize_sentence(sentence, lemmatizer):
    """Lemmatize the sentence"""
    tokens = word_tokenize(sentence.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


def replace_target_word(sentence, target_word, replacement_word):
    """Replace target word with replacement word in sentence if not already present."""
    pattern = r'\b' + re.escape(target_word) + r'\b'
    if re.search(pattern, sentence, flags=re.IGNORECASE) and replacement_word not in sentence:
        return re.sub(pattern, replacement_word, sentence, flags=re.IGNORECASE)
    return sentence

def preprocess_sentences(file, output):
    """Filter out rows where target words are not found in their respective sentences"""
    df = pd.read_csv(file) 

    filtered_rows = []
    lemmatizer = WordNetLemmatizer()

    for idx, row in df.iterrows():
        target_word1 = extract_target_word(row['synset1'])
        target_word2 = extract_target_word(row['synset2'])

        # Skip rows where target words are None or identical
        if target_word1 is None or target_word2 is None or target_word1 == target_word2:
            continue

        lemmatized_sentence1 = lemmatize_sentence(row['sentence1'], lemmatizer)
        lemmatized_sentence2 = lemmatize_sentence(row['sentence2'], lemmatizer)

        # check if target words are present in the lemmatized sentences
        if target_word1 in lemmatized_sentence1 and target_word2 in lemmatized_sentence2:
            filtered_rows.append(row)
        else:
            print("!!! ",row['id'] )
    filtered_df = pd.DataFrame(filtered_rows, columns=df.columns)
    filtered_df.to_csv(output)

def modify_sentences(input_csv, output_valid_csv, output_problematic_csv):

    df = pd.read_csv(input_csv)

    lemmatizer = WordNetLemmatizer()

    valid_rows = []
    problematic_rows = []

    for idx, row in df.iterrows():
        # extract and lemmatize target words
        target_word1 = extract_target_word(row['synset1'])
        target_word2 = extract_target_word(row['synset2'])

        if target_word1 is None or target_word2 is None:
            problematic_rows.append(row)
            continue

        # lemmatize the sentences
        lemmatized_sentence1 = lemmatize_sentence(row['sentence1'], lemmatizer)
        lemmatized_sentence2 = lemmatize_sentence(row['sentence2'], lemmatizer)

        # synset1 is more specific
        if row['specific'] == 0:  
            general_sentence = lemmatized_sentence2
            if target_word2 in general_sentence:
                # replace the target word in the general sentence with the more specific target word 
                modified_sentence = replace_target_word(general_sentence, target_word2, target_word1)
                df.loc[idx, 'sentence1'] = modified_sentence

                # remove rows where modified sentences are the same
                if modified_sentence != lemmatized_sentence2:
                    valid_rows.append(df.loc[idx])
                else:
                    print(f"Row {idx} removed: Modified sentence1 is the same as sentence2 after odifications.")
                    problematic_rows.append(row)

        # synset2 is more specific
        else:  
            general_sentence = lemmatized_sentence1
            if target_word1 in general_sentence:
                # replace the target word in the general sentence with the more specific target word
                modified_sentence = replace_target_word(general_sentence, target_word1, target_word2)
                df.loc[idx, 'sentence2'] = modified_sentence

                # remove rows where modified sentences are the same
                if modified_sentence != lemmatized_sentence1:
                    valid_rows.append(df.loc[idx])
                else:
                    print(f"Row {idx} removed: Modified sentence2 is the same as sentence1 after modifications.")
                    problematic_rows.append(row)
                

    valid_df = pd.DataFrame(valid_rows, columns=df.columns)
    problematic_df = pd.DataFrame(problematic_rows, columns=df.columns)
    valid_df.to_csv(output_valid_csv, index=False)
    problematic_df.to_csv(output_problematic_csv, index=False)

    print(f"Valid data has been saved to {output_valid_csv}.")
    print(f"Problematic data has been saved to {output_problematic_csv}.")

############## SAVE AND LOAD ###################

def save_data_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'synset1', 'depth1', 'sentence1', 'synset2', 'depth2', 'sentence2', 'specific'])
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)

def load_data_from_csv(file_path, test_size=0.2, val_size=0.1):
    data = pd.read_csv(file_path)
     
    # split the data into train and test sets with stratification
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, stratify=data['specific'])
    
    # split the training data into training and validation sets 
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=18, stratify=train_data['specific'])
    
    return train_data, val_data, test_data

if __name__ == "__main__":

    # adjectives and adverbs don't have hypornym relations, so we don't include them
    '''
    n 8742
    v 9691
    a 4355
    r 3192
    '''
    
    # download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

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
    
    raw_file_path = 'data/raw_synsets.csv'
    balanced = 'data/balanced_raw_synsets.csv'
    
    # balance the dataset
    balance_data(raw_file_path, balanced )
    
    # load and split the data
    train_data, val_data, test_data = load_data_from_csv(balanced)
    

    file = 'data/balanced_raw_synsets.csv'
    filtered_file = 'data/filtered_raw_synsets.csv'
    
    preprocess_sentences(file, filtered_file)
    modify_sentences(filtered_file, 'data/synsets_same_context.csv', 'data/problem.csv')
    
    
