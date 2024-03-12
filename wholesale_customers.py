# Part 2: Cluster Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file ,delimiter=',')
	df.drop(['Channel', 'Region'], axis=1, inplace=True)
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	 # Compute summary statistics
    summary_df = df.describe().transpose()
    # Select the required statistics and round them to the closest integer
    summary_stats = summary_df[['mean', 'std', 'min', 'max']].round().astype(int)
    # Rename columns to lowercase as specified
    summary_stats.columns = ['mean', 'std', 'min', 'max']
    return summary_stats

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def clustering_score(X, y):
    """Calculate the silhouette score for the clustering."""
    return silhouette_score(X, y)

def standardize(df):
    """Standardize the DataFrame."""
    scaler = StandardScaler()
    standardized_df = scaler.fit_transform(df)
    return pd.DataFrame(standardized_df, columns=df.columns, index=df.index)

def kmeans(df, k, init='k-means++'):
    """Apply KMeans clustering to the DataFrame."""
    model = KMeans(n_clusters=k, init=init, n_init=10, random_state=42)
    model.fit(df)
    return pd.Series(model.labels_, index=df.index)

def agglomerative(df, k):
    """Apply Agglomerative clustering to the DataFrame."""
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(df)
    return pd.Series(model.labels_, index=df.index)

def cluster_evaluation(df):
    results = []
    max_k = min(len(df), 10)  # Ensures k does not exceed the number of samples
    for algo in ['Kmeans', 'Agglomerative']:
        for data_type, data_function in [('Original', lambda x: x), ('Standardized', standardize)]:
            data = data_function(df)
            for k in range(2, max_k+1):  # Starts from 2, as 1 cluster is trivial
                if algo == 'Kmeans':
                    labels = kmeans(data, k)
                else:  # Agglomerative
                    labels = agglomerative(data, k)
                score = clustering_score(data, labels)
                results.append({
                    'Algorithm': algo,
                    'data': data_type,
                    'k': k,
                    'Silhouette Score': score
                })
    return pd.DataFrame(results)


def best_clustering_score(rdf):
    """Find the best silhouette score from the results."""
    return rdf['Silhouette Score'].max()

def scatter_plots(df, k=3):
    """Generate scatter plots for each pair of attributes in the DataFrame."""
    standardized_df = standardize(df)
    labels = kmeans(standardized_df, k)
    pd.plotting.scatter_matrix(standardized_df, c=labels, figsize=(10, 10), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
    plt.show()