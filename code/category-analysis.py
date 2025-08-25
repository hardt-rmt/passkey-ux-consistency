import pandas as pd


def load_data():
    """
    Loads data
    :return: dataframe
    """
    return pd.read_excel("../data/analyze.xlsx")


def get_category_cluster_results(df):
    """
    Calculate the distribution of each category across clusters
    """
    category_cluster_dist = df.groupby(['Category', 'Cluster']).size().unstack(fill_value=0)
    category_totals = category_cluster_dist.sum(axis=1)
    category_cluster_percent = category_cluster_dist.div(category_totals, axis=0) * 100
    category_cluster_percent = category_cluster_percent.round(0)
    category_cluster_percent.to_excel('../output/category-analysis-results.xlsx')
    print("Category analysis done!")


def main():
    df = load_data()
    get_category_cluster_results(df)


if __name__ == '__main__':
    main()
