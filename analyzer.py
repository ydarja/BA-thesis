from data_manager import load_data_from_csv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import statsmodels.api as sm
from model import SpecificityModel, load_model
from data_manager import load_data_from_csv
import matplotlib.pyplot as plt
import MoreThanSentiments as mts

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence1 = self.df.iloc[idx]['sentence1']
        sentence2 = self.df.iloc[idx]['sentence2']
        depth1 = self.df.iloc[idx]['depth1']
        depth2 = self.df.iloc[idx]['depth2']

        encoding1 = self.tokenizer(sentence1, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        encoding2 = self.tokenizer(sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'depth1': torch.tensor(depth1, dtype=torch.float),
            'depth2': torch.tensor(depth2, dtype=torch.float),
            'sentence1': sentence1,
            'sentence2': sentence2
        }

def get_specificity_scores(test_df, model_path='models/model5', batch_size=4, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = TestDataset(test_df, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, epoch = load_model(model, optimizer, model_path, device)
    model.eval()

    specificity_scores, true_depths, test_sentences, sentence_lengths = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            depth1 = batch['depth1'].to(device)
            depth2 = batch['depth2'].to(device)
                
            logits1, logits2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                
            scores1 = logits1.squeeze().cpu().numpy()
            scores2 = logits2.squeeze().cpu().numpy()
                
            for idx in range(len(batch['sentence1'])):
                sentence1 = batch['sentence1'][idx]
                sentence2 = batch['sentence2'][idx]

                sentence_lengths.append(len(sentence1.split()))
                sentence_lengths.append(len(sentence2.split()))

                test_sentences.extend([sentence1, sentence2])
                specificity_scores.extend([scores1[idx], scores2[idx]])
                true_depths.extend([depth1[idx].cpu().numpy(), depth2[idx].cpu().numpy()])

    specificity_scores = np.array(specificity_scores)
    # normalize  -> [0,1]
    scaler = MinMaxScaler()
    specificity_scores = scaler.fit_transform(specificity_scores.reshape(-1, 1)).flatten()
    true_depths = np.array(true_depths)
    sentence_lengths = np.array(sentence_lengths)

    return specificity_scores, true_depths, test_sentences, sentence_lengths

def analyze(test_df):
    specificity_scores, true_depths, test_sentences, sentence_lengths = get_specificity_scores(test_df)
    
    # correlation between specificity scores and true depths
    pearson_corr_depth, _ = pearsonr(specificity_scores, true_depths)
    spearman_corr_depth, _ = spearmanr(specificity_scores, true_depths)
    
    # correlation between specificity scores and sentence lengths
    pearson_corr_length, _ = pearsonr(specificity_scores, sentence_lengths)
    spearman_corr_length, _ = spearmanr(specificity_scores, sentence_lengths)
    
    print(f'Pearson correlation (Specificity & Depth): {pearson_corr_depth}')
    print(f'Spearman correlation (Specificity & Depth): {spearman_corr_depth}')
    
    print(f'Pearson correlation (Specificity & Sentence Length): {pearson_corr_length}')
    print(f'Spearman correlation (Specificity & Sentence Length): {spearman_corr_length}')
    
    # descriptive Statistics
    print(f'Mean specificity score: {np.mean(specificity_scores)}')
    print(f'Median specificity score: {np.median(specificity_scores)}')
    print(f'Standard deviation of specificity scores: {np.std(specificity_scores)}')
    print(f'Mean WordNet depth: {np.mean(true_depths)}')
    print(f'Median WordNet depth: {np.median(true_depths)}')
    print(f'Standard deviation of WordNet depths: {np.std(true_depths)}')

    # distribution Plot
    plt.figure(figsize=(8, 6))
    plt.hist(specificity_scores, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Specificity Scores', fontsize=14)
    plt.grid(True)
    plt.savefig('new/specificity_score_distribution.png', dpi=300)
    plt.close()

    # regression Analysis
    X = sm.add_constant(specificity_scores)
    regression_model = sm.OLS(true_depths, X).fit()
    print("Regression Analysis")
    print(regression_model.summary())

    # scatter Plot of Specificity vs True Depths
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, true_depths)
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Scatter Plot of Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    plt.savefig('new/scatter_plot.png', dpi=300)
    plt.close()

    # scatter Plot of Specificity vs Sentence Lengths
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, sentence_lengths)
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('Sentence Lengths', fontsize=12)
    plt.title('Scatter Plot of Specificity Scores vs Sentence Lengths', fontsize=14)
    plt.grid(True)
    plt.savefig('new/scatter_plot_lengths.png', dpi=300)
    plt.close()

    # regression Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, true_depths, label='Data points')
    plt.plot(specificity_scores, regression_model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Regression Analysis: Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig('new/regression_plot.png', dpi=300)
    plt.close()

    # boxplots by Depth
    specific_depths = [2, 6, 10, 14]
    for depth in specific_depths:
        mask = (true_depths == depth)
        filtered_scores = specificity_scores[mask]
        filtered_depths = true_depths[mask]
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_scores, vert=False)
        plt.xlim(0, 1)
        plt.xlabel('Specificity Scores', fontsize=12)
        plt.ylabel('True Depths', fontsize=12)
        plt.title(f'Specificity Scores at WordNet Depth {depth}', fontsize=14)
        plt.grid(True)
        plt.savefig(f'new/boxplot_depth{depth}.png', dpi=300)
        plt.show()
        plt.close()

    # comparison with MTS
    other_specificity_scores = mts.Specificity(test_sentences)
    correlation_with_other, _ = pearsonr(specificity_scores, other_specificity_scores)
    print(f'Correlation with MoreThanSentiments: {correlation_with_other}')

    plt.figure(figsize=(10, 6))
    plt.scatter(specificity_scores, other_specificity_scores, color='blue', label='Data points', s=50, alpha=0.6)
    coefficients = np.polyfit(specificity_scores, other_specificity_scores, 1)
    polynomial = np.poly1d(coefficients)
    regression_line = polynomial(specificity_scores)
    plt.plot(specificity_scores, regression_line, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('WordNet-based Specificity Scores')
    plt.ylabel('MoreThanSentiments Specificity Scores')
    plt.title('Comparison of Specificity Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('new/morethansentiments.png', dpi=300)
    plt.close()

def calculate_sentence_stats(csv_file):
    df = pd.read_csv(csv_file)

    sentence1_lengths = df['sentence1'].apply(lambda x: len(x.split()))
    sentence2_lengths = df['sentence2'].apply(lambda x: len(x.split()))
    combined_lengths = pd.concat([sentence1_lengths, sentence2_lengths])

    # statistics
    mean_length = combined_lengths.mean()
    median_length = combined_lengths.median()
    std_dev_length = combined_lengths.std()

    print(f'Mean Sentence Length: {mean_length}')
    print(f'Median Sentence Length: {median_length}')
    print(f'Standard Deviation of Sentence Lengths: {std_dev_length}')

if __name__ == "__main__":
    file = 'data/balanced_raw_synsets.csv'
    _, _, test = load_data_from_csv(file)
    analyze(test)
    #calculate_sentence_stats(file)
