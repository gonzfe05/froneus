'''
Contains a set of utilities used in the project for plotting features.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.features import RadViz, ParallelCoordinates, PCADecomposition, JointPlotVisualizer

def plot_RadViz(X,labels):
    visualizer = RadViz(alpha=0.3,colormap="prism", size=(1080, 720))
    visualizer.fit(X, labels)
    visualizer.transform(X)
    visualizer.poof()

def plot_PCA(X,labels):
    colors_ = ["b","g","r","c","m","y","k","w"]
    visualizer = PCADecomposition(scale=True, color=[colors_[i] for i in labels.values], proj_features=False, alpha=0.5, size=(900, 600))
    visualizer.fit_transform(X,labels)
    visualizer.poof()
    
def plot_corr(X):
    # Features correlation
    features_corr = X.corr()
    # Show only upper triangle
    upper_mask = np.zeros_like(features_corr, dtype=np.bool)
    upper_mask[np.triu_indices_from(upper_mask)] = True
    # figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Collor pallete
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw
    sns.heatmap(features_corr, mask=upper_mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})