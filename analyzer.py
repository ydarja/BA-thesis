
from data_manager import load_data_from_csv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import statsmodels.api as sm
from model import SpecificityModel, load_model
from data_manager import load_data_from_csv
import matplotlib.pyplot as plt
import MoreThanSentiments as mts

class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence1 = self.df.iloc[idx]['sentence1']
        sentence2 = self.df.iloc[idx]['sentence2']
        depth1 = self.df.iloc[idx]['depth1']
        depth2 = self.df.iloc[idx]['depth2']

        encoding1 = self.tokenizer(sentence1, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        encoding2 = self.tokenizer(sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        item = {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'depth1': torch.tensor(depth1, dtype=torch.float),
            'depth2': torch.tensor(depth2, dtype=torch.float),
            'sentence1': sentence1,
            'sentence2': sentence2
        }
        return item

def get_specificity_scores(test_df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = TestDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, epoch = load_model(model, optimizer, 'models/model1', device)
    model.eval()

    specificity_scores = []
    true_depths = []
    test_sentences = []

    with torch.no_grad():
        for batch in test_loader:
            for idx in range(len(batch['sentence1'])):
                sentence1 = batch['sentence1'][idx]
                sentence2 = batch['sentence2'][idx]
                test_sentences.append(sentence1)
                test_sentences.append(sentence2)
        
                input_ids1 = batch['input_ids1'].to(device)
                attention_mask1 = batch['attention_mask1'].to(device)
                input_ids2 = batch['input_ids2'].to(device)
                attention_mask2 = batch['attention_mask2'].to(device)
                depth1 = batch['depth1'].to(device)
                depth2 = batch['depth2'].to(device)
                
                logits1, logits2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                
                scores1 = logits1.squeeze().cpu().numpy()
                scores2 = logits2.squeeze().cpu().numpy()
                specificity_scores.extend(scores1)
                specificity_scores.extend(scores2)
                true_depths.extend(depth1.cpu().numpy())
                true_depths.extend(depth2.cpu().numpy())

    specificity_scores = np.array(specificity_scores)
    true_depths = np.array(true_depths)
    

    return specificity_scores, true_depths, test_sentences

def analyze(test_df):
    specificity_scores, true_depths, test_sentences = get_specificity_scores(test_df) 
    # Calculate correlations
    pearson_corr, _ = pearsonr(specificity_scores, true_depths)
    spearman_corr, _ = spearmanr(specificity_scores, true_depths)

    print(f'Pearson correlation: {pearson_corr}')
    print(f'Spearman correlation: {spearman_corr}')

    # Descriptive Statistics
    print(f'Mean specificity score: {np.mean(specificity_scores)}')
    print(f'Median specificity score: {np.median(specificity_scores)}')
    print(f'Standard deviation of specificity scores: {np.std(specificity_scores)}')
    print(f'Mean WordNet depth: {np.mean(true_depths)}')
    print(f'Median WordNet depth: {np.median(true_depths)}')
    print(f'Standard deviation of WordNet depths: {np.std(true_depths)}')

    # Regression Analysis
    X = sm.add_constant(specificity_scores)
    regression_model = sm.OLS(true_depths, X).fit()
    print("Regression Analysis")
    print(regression_model.summary())

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, true_depths)
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Scatter Plot of Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    plt.savefig('plots/scatter_plot.png', dpi=300)

    # Regression plot
    plt.figure(figsize=(8, 6))
    plt.plot(specificity_scores, regression_model.predict(X), color='red')
    plt.scatter(specificity_scores, true_depths)
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Regression Analysis: Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    plt.savefig('plots/regression_plot.png', dpi=300)

    # Boxplots by depth plot

    specific_depths = [2,6,10,14]

    for depth in specific_depths:
        mask = (true_depths==depth)
        filtered_scores = specificity_scores[mask]
        filtered_depths = true_depths[mask]
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_scores, vert=False)
        plt.xlabel('Specificity Scores', fontsize=12)
        plt.ylabel('True Depths', fontsize=12)
        plt.title(f'Specificity Scores at WordNet Depth {depth}', fontsize=14)
        plt.grid(True)
        plt.savefig(f'plots/boxplot_depth{depth}.png', dpi=300)
        plt.show()


    # Comparison with OtherThanSentiments
    #other_specificity_scores = mts.Specificity(test_sentences)
    #correlation_with_other = pearsonr(specificity_scores, other_specificity_scores)
    #print(f'Correlation with other library\'s specificity scores: {correlation_with_other}')

if __name__ == "__main__":
    _, _, test_df = load_data_from_csv('data/balanced_raw_synsets.csv')
    analyze(test_df)
