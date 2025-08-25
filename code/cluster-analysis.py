import pandas as pd
import math


def load_data():
    """
    Load the data
    """
    df = pd.read_excel('../data/analyze.xlsx')
    # Identify non-feature columns
    non_feature_cols = ['Websites', 'Cluster', 'Ranking', 'URL', 'Category']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    return df, feature_cols


def get_cluster_analysis_results(df, feature_cols):
    """
    Get deployment rate per feature for each cluster
    :param df:data
    :param feature_cols:comparison factors
    """
    # Prepare a dictionary to store full match percentages
    full_match_table = {}
    for cluster_id, group in df.groupby('Cluster'):
        full_match_percents = []
        for feature in feature_cols:
            # Count full matches (score=3) and partial matches (score=2)
            # for this feature in this cluster
            is_full_match = group[feature] == 3
            is_partial_match = group[feature] == 2
            match = is_full_match.sum() + is_partial_match.sum()
            percent_full_match = match / len(group) * 100
            full_match_percents.append(math.ceil(percent_full_match))
        full_match_table[cluster_id] = full_match_percents

    # Build DataFrame
    full_match_df = pd.DataFrame(full_match_table, index=feature_cols)
    full_match_df.columns = [f'Cluster {c} %' for c in full_match_df.columns]
    full_match_df.index.name = 'Feature'
    full_match_df.to_excel('../output/cluster-analysis-results.xlsx')
    print("Cluster analysis done!")


def main():
    df, feature_cols = load_data()
    get_cluster_analysis_results(df, feature_cols)


if __name__ == '__main__':
    main()
