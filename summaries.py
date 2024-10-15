import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
from model import SpecificityModel, load_model
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, gaussian_kde


def tokenize_sentences(sentences, tokenizer, max_length=128):
    encodings = [tokenizer(sentence, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt') for sentence in sentences]
    input_ids = torch.stack([encoding['input_ids'].squeeze(0) for encoding in encodings])
    attention_masks = torch.stack([encoding['attention_mask'].squeeze(0) for encoding in encodings])
    return input_ids, attention_masks

def get_specificity_scores(sentences, model, tokenizer, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenize sentences
    input_ids, attention_masks = tokenize_sentences(sentences, tokenizer, max_length)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        logits = []
        for i in range(len(sentences)):
            input_ids1 = input_ids[i].unsqueeze(0)  
            attention_mask1 = attention_masks[i].unsqueeze(0)  
            logits1, _ = model(input_ids1=input_ids1, attention_mask1=attention_mask1,
                               input_ids2=input_ids1, attention_mask2=attention_mask1)
            logits.append(logits1.squeeze().cpu().numpy())

    specificity_scores = np.array(logits).flatten()

    return specificity_scores

def compute_specificity_measures(scores):
    """
    Computes different statistical measures from the list of specificity scores.
    """
    if len(scores) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # Handle empty lists

    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    median_score = np.median(scores)
    range_score = np.max(scores) - np.min(scores)
    std_dev = np.std(scores)

    return avg_score, min_score, max_score, median_score, range_score, std_dev

def process_csv(input_csv, output_csv, model_path='models/model5', max_length=128):

    df = pd.read_csv(input_csv)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load the model
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_model(model, optimizer, model_path, device)
    model.eval()

    all_scores = []
    # collect raw specificity scores for each summary
    for idx, row in df.iterrows():
        summary_a = row['summary_a']
        summary_b = row['summary_b']
        
        # split summaries into sentences
        sentences_a = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_a)
        sentences_b = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_b)
        
        # calculate raw specificity scores
        scores_a = get_specificity_scores(sentences_a, model, tokenizer, max_length)
        scores_b = get_specificity_scores(sentences_b, model, tokenizer, max_length)

        all_scores.extend(scores_a)
        all_scores.extend(scores_b)
    
    # normalize all collected scores globally
    all_scores = np.array(all_scores).reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(all_scores).flatten()


    score_idx = 0

    #  update the dataframe with normalized scores 
    for idx, row in df.iterrows():
        summary_a = row['summary_a']
        summary_b = row['summary_b']
        
        sentences_a = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_a)
        sentences_b = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_b)

        # get the corresponding normalized scores for summary_a and summary_b
        scores_a = normalized_scores[score_idx:score_idx + len(sentences_a)]
        score_idx += len(sentences_a)
        scores_b = normalized_scores[score_idx:score_idx + len(sentences_b)]
        score_idx += len(sentences_b)
        
        # try out different measures to use as specificity score for a text
        avg_score_a, min_score_a, max_score_a, median_score_a, range_score_a, std_dev_a = compute_specificity_measures(scores_a)
        avg_score_b, min_score_b, max_score_b, median_score_b, range_score_b, std_dev_b = compute_specificity_measures(scores_b)
        
        # add the measures to the dataframe
        df.at[idx, 'avg_score_a'] = round(avg_score_a, 2)
        df.at[idx, 'min_score_a'] = round(min_score_a, 2)
        df.at[idx, 'max_score_a'] = round(max_score_a, 2)
        df.at[idx, 'median_score_a'] = round(median_score_a, 2)
        df.at[idx, 'range_score_a'] = round(range_score_a, 2)
        df.at[idx, 'std_dev_a'] = round(std_dev_a, 2)
        
        df.at[idx, 'avg_score_b'] = round(avg_score_b, 2)
        df.at[idx, 'min_score_b'] = round(min_score_b, 2)
        df.at[idx, 'max_score_b'] = round(max_score_b, 2)
        df.at[idx, 'median_score_b'] = round(median_score_b, 2)
        df.at[idx, 'range_score_b'] = round(range_score_b, 2)
        df.at[idx, 'std_dev_b'] = round(std_dev_b, 2)
    
    # reorder columns
    df = df[['id', 'summary_a', 'avg_score_a', 'min_score_a', 'max_score_a', 'median_score_a', 'range_score_a', 'std_dev_a',
             'summary_b', 'avg_score_b', 'min_score_b', 'max_score_b', 'median_score_b', 'range_score_b', 'std_dev_b']]
    
    df.to_csv(output_csv, index=False)


def remove_intro_sentences(summary):
    """
    Removes the first sentence of a summary if it matches common  AI introductory patterns.
    """
    pattern = r"^Here.*?:\s*"
    if re.match(pattern, summary.strip(), re.IGNORECASE):
        summary = re.sub(pattern, '', summary.strip(), count=1)
    return summary

def process_intro_summaries(input_csv):
    """
    Processes the CSV file to remove introductory sentences from summary_a and summary_b.
    """
    df = pd.read_csv(input_csv)
    df['summary_a'] = df['summary_a'].apply(remove_intro_sentences)
    df['summary_b'] = df['summary_b'].apply(remove_intro_sentences)
    df.to_csv(input_csv, index=False)


def analyze_scores(csv_file, measure="avg_score"):
    """
    Analyzes specificity scores for two summaries and compares them based on different measures.
    """
    df = pd.read_csv(csv_file)

    #  mean of the selected measure for both scores a and b
    avg_score_a = df[measure + "_a"].mean()
    avg_score_b = df[measure + "_b"].mean()

    # check if score_a is always greater than score_b
    df['score_a_greater'] = df[measure + "_a"] > df[measure + "_b"]
    
    always_greater = df['score_a_greater'].all()
    percentage_greater = df['score_a_greater'].mean() * 100

    # summary
    print(f"Average {measure}_a: {avg_score_a:.2f}")
    print(f"Average {measure}_b: {avg_score_b:.2f}")
    print(f"Is {measure}_a always greater than {measure}_b? {'Yes' if always_greater else 'No'}")
    print(f"Percentage of rows where {measure}_a > {measure}_b: {percentage_greater:.2f}%")

    # plot
    plt.figure(figsize=(12, 6))
    plt.hist(df[measure + '_a'], bins=30, alpha=0.5, color='blue', label=f'{measure}_a')
    plt.hist(df[measure + '_b'], bins=30, alpha=0.5, color='red', label=f'{measure}_b')
    plt.title(f'Distribution of {measure} Scores')
    plt.xlabel(f'{measure} Score')
    plt.ylabel('Count') 
    plt.legend()

    plt.savefig(f'epoch2/_norm_distribution_{measure}.png', dpi=300)  
    plt.close()

    # paired scatter plot 
    plt.figure(figsize=(8, 8))
    plt.scatter(df[measure + '_a'], df[measure + '_b'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  
    plt.xlabel(f'{measure}_a')
    plt.ylabel(f'{measure}_b')
    plt.title(f'Paired {measure} Scores')
    plt.savefig(f'epoch2/paired_{measure}_ab.png', dpi=300)  
    plt.close()

    # paired t-test 
    t_stat, p_val = ttest_rel(df[measure + "_a"], df[measure + "_b"])
    print(f"Paired t-test for {measure}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

    # Wilcoxon signed-rank test
    w_stat, p_val_wilcoxon = wilcoxon(df[measure + "_a"], df[measure + "_b"])
    print(f"Wilcoxon signed-rank test for {measure}: statistic = {w_stat:.4f}, p-value = {p_val_wilcoxon:.4f}")

def plot_difference_density(input_csv):
    df = pd.read_csv(input_csv)
    df['difference'] = df['score_a'] - df['score_b']
    differences = df['difference'].dropna()
    kde = gaussian_kde(differences, bw_method='scott')
    x = np.linspace(differences.min() - 0.1, differences.max() + 0.1, 1000)
    kde_values = kde.evaluate(x)
    plt.hist(differences, bins=30, density=True, alpha=0.5, color='purple', edgecolor='black', label='Histogram')
    plt.plot(x, kde_values, color='green', alpha=0.7, label='KDE')
    plt.title('Density Plot of Differences between score_a and score_b')
    plt.xlabel('Difference (score_a - score_b)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('epoch2/difference_density.png', dpi=300)
    plt.close()


def inspect_outliers(input_csv):
    df = pd.read_csv(input_csv)
    pd.set_option('display.max_columns', None) 

    print("Basic statistics for score_a:")
    print(df['score_a'].describe())

    # a threshold for outlier detection (manually chosen)
    threshold = 0.01
    outliers = df[df['score_a'] < threshold]
    print("\nOutliers in score_a:")
    print(outliers)

    df_cleaned = df[df['score_a'] >= threshold]

    print("Basic statistics for score_a without outliers:")
    print(df_cleaned['score_a'].describe())

    df_cleaned.to_csv(input_csv, index=False)


def wtf(input_csv):
    '''
    A function to inspect the top 5 cases
    where score_b is bigger than score_a
    '''
    df = pd.read_csv(input_csv)
    pd.set_option('display.max_columns', None) 
    df['difference'] = df['score_b'] - df['score_a']
    df_higher_b = df[df['difference'] > 0]
    df_higher_b_sorted = df_higher_b.sort_values(by='difference', ascending=False)
    print("Top 5 instances where score_b is higher than score_a:")
    return df_higher_b_sorted.head(5).iloc[:, 0].tolist()

def add_title(input_csv1, input_csv2, output_csv):
    """
    Adds title of articles, difference between score_a and score_b,
    more_specific binary value, annotatotor_id and anotator_choice 
    columns to the dataframe.

    Filters out data points where difference in scores is <= 0.1
    """
    df1 = pd.read_csv(input_csv1)
    df2 = pd.read_csv(input_csv2)

    # add columns
    title = df1['title'].copy()
    df2['title'] = title
    df2 = df2[['id', 'title','summary_a','score_a','summary_b','score_b']]
    df2['difference'] = round(df2['score_a'] - df2['score_b'], 2)
    df2['more_specific'] = df2.apply(lambda row: 0 if row['score_a'] > row['score_b'] else 1, axis=1)
    df2['annotator_id'] = np.nan
    df2['annotator_choice'] = np.nan

    # filter out summaries with small difference
    df2 = df2[abs(df2['difference']) >= 0.1]
    df2 = df2.sample(frac=1)
    df2.to_csv(output_csv)

def calculate_average_text_length(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'text' not in df.columns:
        raise ValueError("The CSV file does not contain a 'text' column.")
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    average_length = df['text_length'].mean()
    return average_length

if __name__ == "__main__":
    wiki = 'human_evaluation\wikihow100.csv'
    # original mode, 5 epochs
    file1 = 'human_evaluation/generated_summaries_llama3.1.csv'
    file2 = 'human_evaluation/generated_summaries_llama3.1_2.csv'
    file3 = 'human_evaluation/generated_summaries_llama3.1_3.csv'
    file4 = 'human_evaluation/generated_summaries_llama3.1_4.csv'
    file5 = 'human_evaluation/generated_summaries_gemma2.csv'

    # same context, 2 epochs
    file6 = 'human_evaluation/generated_summaries_llama3.1_4_sc.csv'
    # original model, 2 epochs
    file7 = 'human_evaluation/generated_summaries_llama3.1_4_2epochs.csv'

    out = 'human_evaluation/generated_summaries_llama3.1_4_2epochs_norm.csv'
    process_csv(file7, out)
    analyze_scores(out, measure='std_dev')
    plot_difference_density(file7)
    

    
