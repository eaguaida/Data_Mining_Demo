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
    k_values=[3, 5, 10]
    best_score = -1
    best_labels = None
    results = []

    standardized_df = standardize(df)

    for k in k_values:
        for algo in ['KMeans', 'Agglomerative']:
            if algo == 'KMeans':
                labels = kmeans(standardized_df, k)
            else:  # Agglomerative
                labels = agglomerative(standardized_df, k)
            
            score = clustering_score(standardized_df, labels)
            results.append({
                'Algorithm': algo,
                'k': k,
                'Silhouette Score': score
            })

            if score > best_score:
                best_score = score
                best_labels = labels

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
def scatter_plots(df, labels):
    """
    Generate scatter plots for each pair of attributes in the DataFrame.
    """
    attributes = df.columns
    pairs = list(combinations(attributes, 2))
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (attr1, attr2) in enumerate(pairs):
        ax = axes[i]
        ax.scatter(df[attr1], df[attr2], c=labels, cmap='viridis')
        ax.set_xlabel(attr1)
        ax.set_ylabel(attr2)
        ax.set_title(f"{attr1} vs {attr2}")
    
    plt.tight_layout()
    plt.show()