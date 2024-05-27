# Data Mining Demo

This repository contains the implementation of a Data Mining demostration, which is divided into three examples: decision trees with categorical attributes, cluster analysis, and text mining. Each task involves applying different data mining techniques to specific datasets to extract meaningful insights and build predictive models.

## Part 1: Decision Trees with Categorical Attributes
**Dataset**: [Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult)

### Task
Predict whether the income of an individual exceeds 50K per year based on various attributes.

### Steps
1. Load and preprocess the dataset.
2. Handle missing values and convert categorical attributes using one-hot encoding.
3. Build a decision tree classifier and evaluate its performance.

## Part 2: Cluster Analysis
**Dataset**: [Wholesale Customers dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers)

### Task
Identify similar groups of customers based on annual spending on various product categories.

### Steps
1. Compute summary statistics for the dataset.
2. Perform k-means and hierarchical clustering.
3. Evaluate clustering results using the Silhouette score and visualize the clusters.

## Part 3: Text Mining
**Dataset**: [Coronavirus Tweets NLP dataset](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification)

### Task
Predict the sentiment of tweets related to COVID-19.

### Steps
1. Preprocess the text data by cleaning and tokenizing the tweets.
2. Perform feature extraction using CountVectorizer.
3. Train a Multinomial Naive Bayes classifier and evaluate its performance.
