import pandas as pd


def load_data():
    """
    Load the dataset
    :return: panda dataframe
    """
    return pd.read_excel('../data/analyze.xlsx')


def get_ranking_tiers(df):
    """
    Define ranking tiers
    :param df: data
    :return: ranking tiers
    """
    high = df[(df['Ranking'] >= 1) & (df['Ranking'] < 100)]
    mid = df[(df['Ranking'] >= 100) & (df['Ranking'] < 1001)]
    low = df[df['Ranking'] >= 1000]
    return high, mid, low


def get_cluster_percentages(tier):
    """
    Derive cluster percentages for each group
    :param tier: Ranking tier
    :return: cluster percentages
    """
    cluster_counts = tier['Cluster'].value_counts(normalize=True).sort_index()
    # Ensure all clusters are represented (fill missing with 0)
    all_clusters = [1, 2, 3]  # adjust as needed
    return [cluster_counts.get(c, 0.0) * 100 for c in all_clusters]


def get_rankings_analysis_data(high, mid, low):
    """
    Gather data for each ranking group and
    export to excel
    """
    table_data = [
        get_cluster_percentages(high),
        get_cluster_percentages(mid),
        get_cluster_percentages(low)
    ]

    # Build DataFrame
    ranking_groups = ['High', 'Mid', 'Low']
    clusters = ['Cluster 1 (%)', 'Cluster 2 (%)', 'Cluster 3 (%)']
    table_df = pd.DataFrame(table_data, columns=clusters, index=ranking_groups)
    table_df.to_excel("../output/ranking_analysis_results.xlsx")
    print("Rank analysis done!")


def main():
    dataframe = load_data()
    high_tier, mid_tier, low_tier = get_ranking_tiers(dataframe)
    get_rankings_analysis_data(high_tier, mid_tier, low_tier)


if __name__ == "__main__":
    main()
