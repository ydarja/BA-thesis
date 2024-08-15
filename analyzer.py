
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

def get_specificity_scores(test_df, model_path='models/model3', batch_size=4, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = TestDataset(test_df, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, epoch = load_model(model, optimizer, model_path, device)
    model.eval()

    specificity_scores, true_depths, test_sentences = [], [], []

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
                
                test_sentences.extend([sentence1, sentence2])
                specificity_scores.extend([scores1[idx], scores2[idx]])
                true_depths.extend([depth1[idx].cpu().numpy(), depth2[idx].cpu().numpy()])

    specificity_scores = np.array(specificity_scores)
    # normalize (0,1)
    scaler = MinMaxScaler()
    specificity_scores =  scaler.fit_transform(specificity_scores.reshape(-1, 1)).flatten()
    true_depths = np.array(true_depths)

    return specificity_scores, true_depths, test_sentences


def analyze(test_df):
    specificity_scores, true_depths, test_sentences = get_specificity_scores(test_df)

    '''
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

    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, true_depths)
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Scatter Plot of Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    # SAVE
    plt.savefig('plots/scatter_plot.png', dpi=300)
    plt.close()

    # Regression Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_scores, true_depths, label='Data points')
    plt.plot(specificity_scores, regression_model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Specificity Scores', fontsize=12)
    plt.ylabel('True Depths', fontsize=12)
    plt.title('Regression Analysis: Specificity Scores vs True Depths', fontsize=14)
    plt.grid(True)
    plt.legend()
    # SAVE
    plt.savefig('plots/regression_plot.png', dpi=300)
    plt.close()
    '''
    # Boxplots by Depth

    specific_depths = [2,6,10,14]

    for depth in specific_depths:
        mask = (true_depths==depth)
        filtered_scores = specificity_scores[mask]
        filtered_depths = true_depths[mask]
        plt.figure(figsize=(10, 6))
        plt.boxplot(filtered_scores, vert=False)
        plt.xlim(-10, 11)
        plt.xlabel('Specificity Scores', fontsize=12)
        plt.ylabel('True Depths', fontsize=12)
        plt.title(f'Specificity Scores at WordNet Depth {depth}', fontsize=14)
        plt.grid(True)
        plt.savefig(f'plots/boxplot_depth{depth}.png', dpi=300)
        plt.show()
        plt.close()
    '''
    # Comparison with Other Library
    other_specificity_scores = mts.Specificity(test_sentences)
    correlation_with_other, _ = pearsonr(specificity_scores, other_specificity_scores)
    print(f'Correlation with other library\'s specificity scores: {correlation_with_other}')

    # Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(specificity_scores, other_specificity_scores, color='blue', label='Data points', s=50, alpha=0.6)

    # Fit a line using NumPy
    coefficients = np.polyfit(specificity_scores, other_specificity_scores, 1)
    polynomial = np.poly1d(coefficients)
    regression_line = polynomial(specificity_scores)

    # Plot the regression line
    plt.plot(specificity_scores, regression_line, color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Your Specificity Scores')
    plt.ylabel('Other Library\'s Specificity Scores')
    plt.title('Comparison of Specificity Scores')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('plots/otherthansentiments.png', dpi=300)
    plt.close()
    '''
if __name__ == "__main__":
    _, _, test = load_data_from_csv('data/balanced_raw_synsets.csv')
    analyze(test)
