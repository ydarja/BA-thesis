import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.preprocessing import MinMaxScaler
#from model import SpecificityModel, load_model
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

    # Tokenize sentences
    input_ids, attention_masks = tokenize_sentences(sentences, tokenizer, max_length)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        # We need to pass each sentence pair to the model
        logits = []
        for i in range(len(sentences)):
            input_ids1 = input_ids[i].unsqueeze(0)  # Add batch dimension
            attention_mask1 = attention_masks[i].unsqueeze(0)  # Add batch dimension
            # Use the same sentence for both sides for now
            logits1, _ = model(input_ids1=input_ids1, attention_mask1=attention_mask1,
                               input_ids2=input_ids1, attention_mask2=attention_mask1)
            logits.append(logits1.squeeze().cpu().numpy())

    logits = np.array(logits)

    # Normalize the scores to a range of (0,1)
    scaler = MinMaxScaler()
    specificity_scores = scaler.fit_transform(logits.reshape(-1, 1)).flatten()

    return specificity_scores

def process_csv(input_csv, model_path='models/model3', max_length=128):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_model(model, optimizer, model_path, device)
    model.eval()
    
    # Initialize columns for specificity scores
    df['score_a'] = np.nan
    df['score_b'] = np.nan

    for idx, row in df.iterrows():
        summary_a = row['summary_a']
        summary_b = row['summary_b']
        
        # Split summaries into sentences
        sentences_a = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_a)
        sentences_b = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary_b)
        
        # Calculate specificity scores
        scores_a = get_specificity_scores(sentences_a, model, tokenizer, max_length)
        scores_b = get_specificity_scores(sentences_b, model, tokenizer, max_length)
        
        # Calculate average specificity scores
        avg_score_a = np.mean(scores_a)
        avg_score_b = np.mean(scores_b)
        
        df.at[idx, 'score_a'] = round(avg_score_a, 2)
        df.at[idx, 'score_b'] = round(avg_score_b, 2)
    
    # Reorder columns
    df = df[['id', 'summary_a', 'score_a', 'summary_b', 'score_b']]
    
    # Save the updated DataFrame back to CSV
    df.to_csv(input_csv, index=False)

    
    # Reorder columns
    df = df[['id', 'summary_a', 'score_a', 'summary_b', 'score_b']]
    
    # Save the updated DataFrame back to CSV
    df.to_csv(input_csv, index=False)

def remove_intro_sentences(summary):
    """
    Removes the first sentence of a summary if it matches common introductory patterns.
    
    Parameters:
        text (str): The summary text from which the first sentence should be removed if it matches the pattern.
        
    Returns:
        str: The modified summary without the introductory sentence if it matches the pattern.
    """
    pattern = r"^Here.*?:\s*"
    if re.match(pattern, summary.strip(), re.IGNORECASE):
        summary = re.sub(pattern, '', summary.strip(), count=1)
    return summary

def process_summaries(input_csv):
    """
    Processes the CSV file to remove introductory sentences from summary_a and summary_b if they match specific patterns.
    
    Parameters:
        input_csv (str): Path to the CSV file containing the summaries.
    """
    df = pd.read_csv(input_csv)

    # Apply the function to remove intro sentences from both summary_a and summary_b
    df['summary_a'] = df['summary_a'].apply(remove_intro_sentences)
    df['summary_b'] = df['summary_b'].apply(remove_intro_sentences)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(input_csv, index=False)
    print("Introductory sentences removed where applicable.")

def analyze_scores(csv_file):

    # Load the CSV file
    df = pd.read_csv(csv_file)

    avg_score_a = df['score_a'].mean()
    avg_score_b = df['score_b'].mean()

    # Check if score_a is always greater than score_b
    df['score_a_greater'] = df['score_a'] > df['score_b']
    
    always_greater = df['score_a_greater'].all()
    percentage_greater = df['score_a_greater'].mean() * 100

    # Print summary
    print(f"Average score_a: {avg_score_a:.2f}")
    print(f"Average score_b: {avg_score_b:.2f}")
    print(f"Is score_a always greater than score_b? {'Yes' if always_greater else 'No'}")
    print(f"Percentage of rows where score_a > score_b: {percentage_greater:.2f}%")

    # Plotting distributions of scores
    plt.figure(figsize=(12, 6))

    # Plotting histogram for score_a
    plt.hist(df['score_a'], bins=30, alpha=0.5, color='blue', label='score_a')

    # Plotting histogram for score_b
    plt.hist(df['score_b'], bins=30, alpha=0.5, color='red', label='score_b')

    plt.title('Distribution of Specificity Scores')
    plt.xlabel('Specificity Score')
    plt.ylabel('Count')  # Or use 'Frequency'
    plt.legend()

    plt.savefig('plots/distribution_ab5.png', dpi=300)  
    plt.close()


    # Plotting a paired scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['score_a'], df['score_b'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Identity line
    plt.xlabel('score_a')
    plt.ylabel('score_b')
    plt.title('Paired Specificity Scores')

    plt.savefig('plots/paired_ab5.png', dpi=300)  
    plt.close()

    # Paired t-test
    t_stat, p_val = ttest_rel(df['score_a'], df['score_b'])
    print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_val_wilcoxon = wilcoxon(df['score_a'], df['score_b'])
    print(f"Wilcoxon signed-rank test: statistic = {w_stat:.4f}, p-value = {p_val_wilcoxon:.4f}")

def plot_difference_density(input_csv):
    df = pd.read_csv(input_csv)

    # Calculate the difference between score_a and score_b
    df['difference'] = df['score_a'] - df['score_b']

    # Extract differences for density estimation
    differences = df['difference'].dropna()

    # Create KDE using scipy
    kde = gaussian_kde(differences, bw_method='scott')
    x = np.linspace(differences.min() - 0.1, differences.max() + 0.1, 1000)
    kde_values = kde.evaluate(x)

    plt.figure(figsize=(12, 6))

    # Plot histogram of differences
    plt.hist(differences, bins=30, density=True, alpha=0.5, color='purple', edgecolor='black', label='Histogram')

    # Plot KDE line
    plt.plot(x, kde_values, color='green', alpha=0.7, label='KDE')

    plt.title('Density Plot of Differences between score_a and score_b')
    plt.xlabel('Difference (score_a - score_b)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('plots/difference_density5.png', dpi=300)
    plt.close()


def inspect_outliers(input_csv):
    df = pd.read_csv(input_csv)
    pd.set_option('display.max_columns', None) 

    # Print basic statistics to identify potential outliers
    print("Basic statistics for score_a:")
    print(df['score_a'].describe())

    # Define a threshold for outlier detection (e.g., values close to 0)
    threshold = 0.01
    outliers = df[df['score_a'] < threshold]

    # Print outliers for inspection
    print("\nOutliers in score_a:")
    print(outliers)

    # remove outliers
    df_cleaned = df[df['score_a'] >= threshold]

    print("Basic statistics for score_a withoutthe outlier:")
    print(df_cleaned['score_a'].describe())

    df_cleaned.to_csv(input_csv, index=False)


def wtf(input_csv):
    '''
    A function to inspect the top 5 cases
    where score_b is bigger than score_a
    '''
    df = pd.read_csv(input_csv)
    pd.set_option('display.max_columns', None) 
    
    # Calculate the difference between score_b and score_a
    df['difference'] = df['score_b'] - df['score_a']
    
    # Filter out rows where score_b is higher than score_a
    df_higher_b = df[df['difference'] > 0]
    
    # Sort by the difference in descending order to get the top differences
    df_higher_b_sorted = df_higher_b.sort_values(by='difference', ascending=False)
    
    # Print the top 5 rows
    print("Top 5 instances where score_b is higher than score_a:")
    return df_higher_b_sorted.head(5).iloc[:, 0].tolist()

if __name__ == "__main__":
    file1 = 'human_evaluation/generated_summaries_llama3.1.csv'
    file2 = 'human_evaluation/generated_summaries_llama3.1_2.csv'
    file3 = 'human_evaluation/generated_summaries_llama3.1_3.csv'
    file4 = 'human_evaluation/generated_summaries_llama3.1_4.csv'
    file5 = 'human_evaluation/generated_summaries_gemma2.csv'

    # process_csv(file5)
    # analyze_scores(file5)
    plot_difference_density(file5)