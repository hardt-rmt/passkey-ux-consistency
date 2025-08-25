import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import seaborn as sns


def load_data():
    """
    Load Excel data
    :return: websites and comparison factors
    """
    df = pd.read_excel("../data/analyze.xlsx")
    # Extract website
    websites = df["Websites"].values
    # Identify factor columns (exclude metadata columns)
    metadata_cols = ['Websites', 'Cluster', 'Ranking', 'URL', 'Category']
    factors = [col for col in df.columns if col not in metadata_cols]
    factors_data = df[factors].to_numpy()
    return websites, factors_data


def adjusted_euclidean_distances(row1, row2):
    """
    Euclidean distance function
    :param row1: Website ordinal value
    :param row2: Website ordinal value
    :return: Euclidean distance
    """
    total = 0
    count = 0
    for a, b in zip(row1, row2):
        # Ignore factors with 'not applicable' values for either website
        if a != 0 and b != 0:
            total += (a - b) ** 2
            count += 1
    # Return Nan if there are no comparable features
    if count == 0:
        return np.nan
    """
    Normalize the sum of squared differences by 
    dividing by the number of valid comparisons 
    (count) to compute the root mean squared 
    difference over valid features, which is often 
    preferable when the number of compared features 
    varies between pairs.
    """
    return np.sqrt(total / count)


def adjusted_euclidean_distances_matrix_and_dataframe(names, features):
    """
    Adjusted Euclidean distance matrix and dataframe function
    :param names: website names
    :param features: comparison factors
    :return: distance matrix and dataframe
    """
    # Compute adjusted distance matrix
    n = len(features)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = adjusted_euclidean_distances(features[i], features[j])

    # Normalize distances between 0 and 1
    max_dist = np.nanmax(dist_matrix)
    normalized_dist_matrix = dist_matrix / max_dist

    # Convert to a labeled dataframe
    dist_df = pd.DataFrame(normalized_dist_matrix, index=names, columns=names)
    return dist_matrix, dist_df


def similarity_measurements(dist_df):
    # Get the pairwise distance matrix
    dists = dist_df.values
    """
    -   Extract the upper triangle of the matrix excluding the diagonal to get all unique pairs
    -   np.triu_indices_from is a NumPy function that returns the indices for the upper triangle 
        of a square matrix, excluding the diagonal (because of k=1).
    -   k=1 means “start one position above the diagonal.”
    -   mask: The result is a tuple of two arrays: the row indices and column indices for all 
        elements in the upper triangle (above the main diagonal).
    """
    mask = np.triu_indices_from(dists, k=1)
    pairwise_distances = dists[mask]

    # Filter out nan values (if any)
    pairwise_distances = pairwise_distances[~np.isnan(pairwise_distances)]

    # Calculate statistics
    mean_dist = np.mean(pairwise_distances)
    median_dist = np.median(pairwise_distances)

    # Calculate mode (rounded for better grouping in continuous data)
    mode_result = stats.mode(pairwise_distances, nan_policy='omit')
    mode_dist = mode_result[0]

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(pairwise_distances, bins=20, color="#4c72b0", edgecolor="white", alpha=0.8)

    # Plot mean, min, max, and mode as vertical lines
    plt.axvline(mean_dist, color='orange', linestyle='--', linewidth=2, label=f"Mean: {mean_dist:.2f}")
    plt.axvline(mode_dist, color='purple', linestyle='-.', linewidth=2, label=f"Mode: {mode_dist:.2f}")
    plt.axvline(median_dist, color='green', linestyle='-.', linewidth=2, label=f"Median: {median_dist:.2f}")

    plt.xlabel("Pairwise Euclidean Distances")
    plt.ylabel("Number of Website Pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../output/euclidean_distances.pdf")


def condensed_distance_matrix(normalized_dist_matrix):
    """
    Converts the 2D matrix into a 1D condensed vector to save space and speed up computation
    :param normalized_dist_matrix: similarity distance matrix
    :return: A condensed vector matrix
    """
    # Convert to condensed form for linkage function
    condensed_dist = squareform(normalized_dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    return linkage_matrix


def website_clusters(linkage_matrix, names):
    """
    Cluster websites based on pairwise similarity
    :param linkage_matrix: a condensed distance matrix
    :param names: website names
    :return: a sheet with website clusters and a pie chart showing the clusters
    """
    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    cluster_df = pd.DataFrame({
        'Website': names,
        'Cluster': cluster_labels
    }).sort_values(by='Cluster')
    output_file = "../output/website_clusters.xlsx"
    cluster_df.to_excel(output_file, index=False)
    print(f"Website Clusters exported to '{output_file}'")


def heatmap(linkage_matrix, normalized_df):
    num_clusters = 3
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    # Build a DataFrame for cluster sizes and sorting
    df_clusters = pd.DataFrame({
        'Website': normalized_df.index,
        'Cluster': cluster_labels
    })
    cluster_size = df_clusters['Cluster'].value_counts().sort_values(ascending=False)
    cluster_order = cluster_size.index.tolist()
    cluster_map = {old: new for new, old in enumerate(cluster_order, start=1)}
    df_clusters['OrderedCluster'] = df_clusters['Cluster'].map(cluster_map)

    # Sort for heatmap ordering
    ordered_df = df_clusters.sort_values(['OrderedCluster', 'Website'])
    ordered_names = ordered_df['Website'].values
    ordered_clusters = ordered_df['OrderedCluster'].values
    ordered_matrix = normalized_df.loc[ordered_names, ordered_names]

    # Generate row/col colors
    palette = sns.color_palette("tab10", len(cluster_order))
    row_colors = [palette[c - 1] for c in ordered_clusters]

    # Create the clustered heatmap without row/col clustering
    sns.set(style="white")
    clusterMap = sns.clustermap(
        ordered_matrix,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="viridis",
        figsize=(22, 22),
        linewidths=0,
        cbar_pos=(1.02, 0.2, 0.03, 0.6)
    )
    clusterMap.ax_col_dendrogram.set_visible(False)
    clusterMap.ax_row_dendrogram.set_visible(False)
    clusterMap.ax_heatmap.set_xticklabels(clusterMap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    clusterMap.ax_heatmap.set_yticklabels(clusterMap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=6)

    clusterMap.savefig("../output/heatmap.pdf", format='pdf', dpi=1200)
    print("Ordered Heatmap of Pairwise Website Similarity Done!")


def main():
    websites, factors = load_data()
    distance_matrix, distance_matrix_df = adjusted_euclidean_distances_matrix_and_dataframe(websites, factors)
    similarity_measurements(distance_matrix_df)
    condensed_matrix = condensed_distance_matrix(distance_matrix_df)
    website_clusters(condensed_matrix, websites)
    heatmap(condensed_matrix, distance_matrix_df)


if __name__ == "__main__":
    main()
