# Part 2: Cluster Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from itertools import combinations
# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    """
    Read Data File
    """
    df = pd.read_csv(data_file ,delimiter=',')
    df.drop(['Channel', 'Region'], axis=1, inplace=True)
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    """
    Compute summary statistics
    """
    summary_df = df.describe().transpose()
    # Select the required statistics and round them to the closest integer
    summary_stats = summary_df[['mean', 'std', 'min', 'max']].round().astype(int)
    # Rename columns to lowercase as specified
    summary_stats.columns = ['mean', 'std', 'min', 'max']
    return summary_stats

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    """
    Standardize the DataFrame.
    """
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)
    return pd.DataFrame(standardized_df, columns=df.columns, index=df.index)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k, init='random'):
    """
    Apply KMeans clustering to the DataFrame.
    """
    model = KMeans(n_clusters=k, init=init, n_init=10, random_state=42)
    model.fit(df)
    return pd.Series(model.labels_, index=df.index)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    """
    Apply KMeans++ clustering to the DataFrame.
    """
    model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    model.fit(df)
    return pd.Series(model.labels_, index=df.index)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    """
    Apply Agglomerative clustering to the DataFrame.
    """
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(df)
    return pd.Series(model.labels_, index=df.index)

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X, y):
    """
    Calculate the silhouette score for the clustering.
    """
    return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    """
    Evaluate clustering using KMeans and Agglomerative clustering over a range of k values.
    Returns a DataFrame of results, the best score, and the corresponding labels.
    """
    k_values = [3, 5, 10]
    best_score = -1
    best_labels = None
    results = []

    # Standardize the data
    standardized_df = standardize(df)

    # Loop over both original and standardized data
    for data_type, data_to_cluster in [('Original', df), ('Standardized', standardized_df)]:
        for k in k_values:
            # Apply KMeans
            kmeans_labels = kmeans(data_to_cluster, k)
            kmeans_score = clustering_score(data_to_cluster, kmeans_labels)
            results.append({
                'Algorithm': 'KMeans',
                'Data Type': data_type,
                'k': k,
                'Silhouette Score': kmeans_score
            })

            if kmeans_score > best_score:
                best_score = kmeans_score
                best_labels = ('KMeans', kmeans_labels)

            # Apply Agglomerative Clustering
            agglomerative_labels = agglomerative(data_to_cluster, k)
            agglomerative_score = clustering_score(data_to_cluster, agglomerative_labels)
            results.append({
                'Algorithm': 'Agglomerative',
                'Data Type': data_type,
                'k': k,
                'Silhouette Score': agglomerative_score
            })

            if agglomerative_score > best_score:
                best_score = agglomerative_score
                best_labels = ('Agglomerative', agglomerative_labels)

    results_df = pd.DataFrame(results)
    return results_df, best_score, best_labels


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    """
    Find the best silhouette score from the results.
    """
    return rdf['Silhouette Score'].max()

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    """
    Generate scatter plots for each pair of attributes in the DataFrame without cluster labels.
    """
    attributes = df.columns
    pairs = list(combinations(attributes, 2))
    
    # Calculate the number of rows and columns for subplot grid
    num_pairs = len(pairs)
    num_cols = 3  # setting 3 columns for the subplots
    num_rows = num_pairs // num_cols + (num_pairs % num_cols > 0)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()
    
    for i, (attr1, attr2) in enumerate(pairs):
        ax = axes[i]
        ax.scatter(df[attr1], df[attr2], alpha=0.6, edgecolor='w')
        ax.set_xlabel(attr1)
        ax.set_ylabel(attr2)
        ax.set_title(f"{attr1} vs {attr2}")
    

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


