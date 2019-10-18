'''
Contains a set of utilities used in the project for clustering.
'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer
from yellowbrick.features import RadViz, ParallelCoordinates, PCADecomposition

def kmeans(X,n_clusters):
	'''
	wrapper for scikit's kmeans
	'''
	if "customer_id" in X.columns:
		X = X.drop("customer_id",axis=1)
	# seed of 10 for reproducibility.
	clusterer = KMeans(n_clusters=n_clusters, random_state=1991)
	cluster_labels = clusterer.fit_predict(X)
	return cluster_labels

def kmeans_selector(X,range_n_clusters):
	if "customer_id" in X.columns:
		X = X.drop("customer_id",axis=1)
	for n_clusters in range_n_clusters:
		cluster_labels = kmeans(X,n_clusters)
		# The silhouette_score gives the average value for all the samples.
		silhouette_avg = silhouette_score(X, cluster_labels)
		print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
		# Compute the silhouette scores for each sample
		model = KMeans(n_clusters,random_state=1991)
		visualizerIntercluster = InterclusterDistance(model)
		visualizerIntercluster.fit(X)
		visualizerIntercluster.poof()
	model = KMeans(random_state=1991)
	visualizerElbow = KElbowVisualizer(model, k=(2,max(range_n_clusters)), metric='silhouette', timings=True)
	visualizerElbow.fit(X)
	visualizerElbow.poof()
