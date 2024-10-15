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


def calculate_kappa(csv_file, min_valid_pairs=3):
    data = pd.read_csv(csv_file)
    batch_ids = data['batch'].unique()
    all_kappa_scores = []

    for batch_id in batch_ids:
        batch_data = data[data['batch'] == batch_id]
        kappa_scores = []

        for i in range(3):  
            for j in range(i + 1, 3):
                valid_annotator1 = []
                valid_annotator2 = []

                for idx, row in batch_data.iterrows():
                    score1 = str(row[f'annotator{i+1}'])
                    score2 = str(row[f'annotator{j+1}'])

                    if score1 in ['0', '1', '-'] and score2 in ['0', '1', '-']:
                        valid_annotator1.append(score1)
                        valid_annotator2.append(score2)

                if len(valid_annotator1) >= min_valid_pairs:
                    # Cohen's Kappa
                    kappa = cohen_kappa_score(valid_annotator1, valid_annotator2, labels=['0', '1', '-'])
                    kappa_scores.append(kappa)
                else:
                    print(f"Skipping kappa calculation for annotators {i+1} and {j+1} in batch {batch_id} due to insufficient valid data.")

        if kappa_scores:
            mean_kappa = np.nanmean(kappa_scores)  
            all_kappa_scores.append(mean_kappa)
            print(f'Batch {batch_id} - Mean Kappa Score: {mean_kappa:.2f}')
        else:
            print(f"No valid kappa scores to calculate for batch {batch_id}.")

    if all_kappa_scores:
        overall_average_kappa = np.nanmean(all_kappa_scores)  # Calculate overall average while ignoring NaN
        print(f'Overall Average Kappa Score: {overall_average_kappa:.2f}')
    else:
        print('No valid kappa scores across all batches to calculate an overall average.')


if __name__ == "__main__":
    file = 'human_evaluation\human_evaluation_final.csv'
    correlations = calculate_point_biserial_correlation(file)
    print(correlations)
    calculate_kappa(file)
