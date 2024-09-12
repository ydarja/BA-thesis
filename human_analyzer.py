import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


def calculate_point_biserial_correlation(csv_file):

    data = pd.read_csv(csv_file)

    annotator_columns = ['annotator1', 'annotator2', 'annotator3']
    all_choices = pd.concat([pd.to_numeric(data[col], errors='coerce') for col in annotator_columns], ignore_index=True)

    specificity_diff_repeated = pd.concat([data['difference']] * 3, ignore_index=True)

    print(len(all_choices))
    print(len(specificity_diff_repeated))

    print("-" * 20)
    # remove entries with NaN values or invalid choices
    valid_data = all_choices.isin([0, 1])

    print(valid_data.value_counts())

    all_choices = all_choices[valid_data].reset_index(drop=True)
    specificity_diff_repeated = specificity_diff_repeated[valid_data].reset_index(drop=True)
    
    print(len(all_choices))
    print(len(specificity_diff_repeated))
    
    correlation, p_value = pointbiserialr(all_choices, specificity_diff_repeated)
    print(f"Correlation: {correlation}, p-value: {p_value}")
    
    # plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(specificity_diff_repeated, all_choices, alpha=0.5)
    plt.xlabel('Specificity Score Difference')
    plt.ylabel('Annotator Choice (0 or 1)')
    plt.title(f'Overall Point-Biserial Correlation\nCorrelation: {correlation:.2f}, p-value: {p_value:.2e}')
    plt.grid(True)
    
    plt.savefig('human_evaluation/combined_correlation.png')
    plt.show()

    return {'correlation': correlation, 'p_value': p_value}


def calculate_kappa_and_plot(csv_file, min_valid_pairs=3):
    data = pd.read_csv(csv_file)
    batch_ids = data['batch'].unique()
    
    for batch_id in batch_ids:
        batch_data = data[data['batch'] == batch_id]
        kappa_scores = []

        for i in range(3):
            for j in range(i + 1, 3):
                valid_annotator1 = []
                valid_annotator2 = []

                for idx, row in batch_data.iterrows():
                    score1 = row[f'annotator{i+1}']
                    score2 = row[f'annotator{j+1}']


                    if score1 in [0, 1] and score2 in [0, 1]:
                        valid_annotator1.append(score1)
                        valid_annotator2.append(score2)

                if len(valid_annotator1) >= min_valid_pairs:
                    kappa = cohen_kappa_score(valid_annotator1, valid_annotator2, labels=[0, 1])
                    kappa_scores.append(kappa)
                else:
                    print(f"Skipping kappa calculation for annotators {i+1} and {j+1} in batch {batch_id} due to insufficient valid data.")

        if kappa_scores:
            plt.figure(figsize=(8, 6))
            sns.histplot(kappa_scores, bins=10, kde=True)
            plt.title(f'Inter-Annotator Agreement (Cohen\'s Kappa) for Batch {batch_id}')
            plt.xlabel('Cohen\'s Kappa Score')
            plt.ylabel('Frequency')
            plt.savefig(f'kappa_partial_batch_{batch_id}.png')
            plt.close()
            print(f'Batch {batch_id} - Mean Kappa Score: {sum(kappa_scores) / len(kappa_scores):.2f}')
        else:
            print(f"No valid kappa scores to plot for batch {batch_id}.")

def count_and_plot_occurrences(csv_file):
    df = pd.read_csv(csv_file)
    
    counts = {'0': 0, '1': 0, '-': 0}
    for column in ['annotator1', 'annotator2', 'annotator3']:
        counts['0'] += (df[column] == '0').sum()
        counts['1'] += (df[column] == '1').sum()
        counts['-'] += (df[column] == '-').sum()


    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of 0, 1, and - in Annotator Columns')
    plt.axis('equal')  
    plt.savefig('epoch2/human_distribution.png')
    plt.close()

if __name__ == "__main__":
    file = 'human_evaluation\human_evaluation_partial.csv'
    # correlations = calculate_point_biserial_correlation(file)
    # print(correlations)
    # calculate_kappa_and_plot(file)
    count_and_plot_occurrences(file)