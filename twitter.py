import pandas as pd
import re
import torch
import numpy as np
from model import SpecificityModel, load_model
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def clean_tweet(tweet):
    tweet = tweet.replace("<USER>", "")
    # remove links
    tweet = tweet.replace("<URL>", "")
    # remove emojis
    tweet = re.sub(r'[^\w\s,.!?]', '', tweet)  
    
    # check if the tweet has more than one sentence
    sentence_count = len(re.findall(r'[.!?]', tweet))  
    if sentence_count > 1:
        return None  
    
    return tweet.strip()

def preprocess_tweets(input_csv):
    df = pd.read_csv(input_csv, sep=',')
    df['Cleaned_tweet'] = df['Tweet'].apply(clean_tweet)
    # filter out tweets with more than one sentence
    df = df[df['Cleaned_tweet'].notna()]
    df.to_csv(input_csv, index=False)
    print("All done! Saved ", len(df), " tweets")

def normalize_twitter_scores(input_csv):
    df = pd.read_csv(input_csv, sep=',', header=0)
    pd.set_option('display.max_columns', None) 
    print(df.head())

    print("Column names:", df.columns.tolist())

    if 'Score' not in df.columns:
        raise ValueError("The dataset does not contain a 'Score' column, please check the column names.")

    
    df['Normalized_score'] = (df['Score'] - 1) / 4
    df.to_csv(input_csv, index=False)
    print(f"Normalization complete. Added 'Normalized_score' column to {input_csv}")

######################################################

def tokenize_sentence(sentence, tokenizer, max_length=128):
    encoding = tokenizer(sentence, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    input_ids = encoding['input_ids'].squeeze(0)  
    attention_mask = encoding['attention_mask'].squeeze(0)  
    return input_ids, attention_mask

def get_specificity_score(sentence, model, tokenizer, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    input_ids, attention_mask = tokenize_sentence(sentence, tokenizer, max_length)
    input_ids = input_ids.unsqueeze(0).to(device)  
    attention_mask = attention_mask.unsqueeze(0).to(device) 

    with torch.no_grad():
        logits, _ = model(input_ids1=input_ids, attention_mask1=attention_mask,
                          input_ids2=input_ids, attention_mask2=attention_mask)
        logits = logits.squeeze().cpu().numpy()

    return logits

def process_tweets(input_csv, model_path='models/model5', max_length=128):

    df = pd.read_csv(input_csv, sep=',')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_model(model, optimizer, model_path, device)
    model.eval()

    logits_list = []
    df['My_score'] = np.nan
    for idx, row in df.iterrows():
        tweet = row['Cleaned_tweet']
        if not tweet or pd.isna(tweet):
            continue
        
        logit_score = get_specificity_score(tweet, model, tokenizer, max_length)
        logits_list.append(logit_score)
        df.at[idx, 'Logits'] = logit_score.item()

    
    scaler = MinMaxScaler()
    df['My_score'] = round(scaler.fit_transform(df[['Logits']]))


    df.to_csv(input_csv, index=False)
    print("Scores have been updated and saved.")

#########################################################

def analyze_logits(csv_file):
    df = pd.read_csv(csv_file, sep=',')
    # Calculate and print statistics
    print("\nLogits Statistics:")
    print(f"Max: {df['Logits'].max()}")
    print(f"Min: {df['Logits'].min()}")
    print(f"Mean: {df['Logits'].mean()}")
    print(f"Standard Deviation: {df['Logits'].std()}")

    # Plot the distribution of logits
    plt.figure(figsize=(12, 6))
    plt.hist(df['Logits'], bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Logits')
    plt.xlabel('Logits')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('epoch2/logits_distribution2.png', dpi=300)  
    plt.close()

def analyze_scores(csv_file):
    df = pd.read_csv(csv_file, sep=',')
    # calculate and print statistics of raw scores
    print("\nMy Score Statistics:")
    print(f"Max: {df['My_score'].max()}")
    print(f"Min: {df['My_score'].min()}")
    print(f"Mean: {df['My_score'].mean()}")
    print(f"Standard Deviation: {df['My_score'].std()}")

    plt.figure(figsize=(12, 6))
    plt.hist(df['My_score'], bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of my Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('epoch2/my_score_distribution2.png', dpi=300)  
    plt.close()

def analyze_twitter_scores(csv_file):
    df = pd.read_csv(csv_file, sep=',')
    # calculate and print statistics of normalized scores
    print("\nNormalized Twitter Score Statistics:")
    print(f"Max: {df['Normalized_score'].max()}")
    print(f"Min: {df['Normalized_score'].min()}")
    print(f"Mean: {df['Normalized_score'].mean()}")
    print(f"Standard Deviation: {df['Normalized_score'].std()}")

    plt.figure(figsize=(12, 6))
    plt.hist(df['Normalized_score'], bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Normalzied Twitter Scores')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('epoch2/norm_twitter_score_distribution2.png', dpi=300)  
    plt.close()

def analyze_correlation(input_csv):
    df = pd.read_csv(input_csv, sep=',')
    correlation = df['My_score'].corr(df['Normalized_score'])
    print(f"Correlation between My Scores andTwitter Scores: {correlation:.2f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(df['My_score'], df['Normalized_score'], alpha=0.5)
    plt.xlabel('My Score')
    plt.ylabel(' Twitter Score')
    plt.savefig('epoch2/paired_twitter.png', dpi=300)  
    plt.close()



if __name__ == "__main__":
    data = 'data/cleaned_data_test3.tsv'
    preprocess_tweets(data)
    normalize_twitter_scores(data)
    process_tweets(data)

    analyze_logits(data)
    analyze_scores(data)
    analyze_correlation(data)
