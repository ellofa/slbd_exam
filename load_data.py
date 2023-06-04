import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer

def load_data():
    #loading the data
    #5000 observations, 1000 features
    data = pd.read_csv('DataMatrix.txt', sep="\t")

    # column 0 contains classes
    data.columns = [str(i) for i in range(1001)]
    data[data == -1000] = np.nan
    # seperate X and y
    X = data.loc[:, data.columns != '0'].to_numpy()
    y = np.ravel(data.loc[:,'0'].to_numpy())

    '''
    #Visualize the data
    # normalize values
    scaler = Normalizer().fit(X)
    normalizedX = scaler.transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)  # Reduce to 2 dimensions for visualization
    X_reduced = pca.fit_transform(normalizedX)

    # Create a scatter plot of the reduced data, with different colors for each class
    unique_labels = np.unique(y)
    colors = ['r', 'g', 'b', 'y']  # Assigning colors for each class
    for label, color in zip(unique_labels, colors):
        indices = y == label
        plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], c=color, label=label)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.legend()
    plt.show()

    unique_labels = np.unique(y)
    colors = ['r', 'g', 'b', 'y']  # Assigning colors for each class
    for label, color in zip(unique_labels, colors):
        indices = y == label
        plt.scatter(X_reduced[indices, 1], X_reduced[indices, 2], c=color, label=label)

    plt.xlabel('Principal Component 2')
    plt.ylabel('Principal Component 3')
    plt.title('PCA Visualization2')
    plt.legend()
    plt.show()
'''

    return X,y,data
