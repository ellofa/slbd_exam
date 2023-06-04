from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFECV
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture


def drop_missing(data):
    data[data == -1000] = np.nan
    data.dropna(inplace=True)
    n = data.shape[0]
    X = data.loc[:, data.columns != '0'].to_numpy()
    y = data.loc[:,'0'].to_numpy().reshape(n,1)
    return X,y,data

def class_balance(y):
    print("nb of samples belonging to class 1", np.sum(y == 1))
    print("nb of samples belonging to class 2", np.sum(y == 2))
    print("nb of samples belonging to class 3", np.sum(y == 3))
    print("nb of samples belonging to class 4", np.sum(y == 4))
    print("nb of samples belonging to class 5", np.sum(y == 5))
    print("nb of samples belonging to class 6", np.sum(y == 6))

def softmax_regression(X_train,y_train,penalty):
    sr = LogisticRegression(solver='lbfgs', max_iter=500, penalty=penalty, multi_class='multinomial')

    # cross validation score
    scores = cross_val_score(sr, X_train, y_train, cv = 10)
    mean = np.mean(scores)
    std = np.std(scores)
    print("mean score of training data with Softmax Regression :", mean)
    print("standard deviation score of training data with Softmax Regression :", std)

    return sr, mean, std, scores

def naive_bayes(X_train,y_train):
    nb = GaussianNB()

    # cross validation score
    scores = cross_val_score(nb, X_train, y_train, cv = 10)
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"mean score of training data with Naive Bayes: {mean}")
    print(f"standard deviation score of training data with Naive Bayes: {std}")

    return nb, mean, std, scores

def knneigh(X_train,y_train):
        clf = KNeighborsClassifier(metric="minkowski")

        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {"n_neighbors": np.arange(4, 10)}
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(clf, param_grid, cv=5)
        #fit model to data
        knn_gscv.fit(X_train, y_train)
        #check top performing n_neighbors value
        k = knn_gscv.best_params_['n_neighbors']
        print("best k-param selected:",k)

        #create the knn model
        knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski")

        # cross validation score
        # train model and calculate cross validation score 
        # number of folds: 10
        # score is the accuracy
        scores = cross_val_score(knn, X_train, y_train, cv = 10)
        mean = np.mean(scores)
        print(f"mean score of training data with {k}-nn :", mean)
        std = np.std(scores)
        print(f"standard deviation score of training data with  {k}-nn: {std}")

        return knn,k,mean,std, scores

def linear_discr(X_train,y_train):

    # create model
    lda = LinearDiscriminantAnalysis()

    # cross validation score
    scores = cross_val_score(lda, X_train, y_train, cv = 10)
    mean = np.mean(scores)
    print(f"mean score of training data with LDA :", mean)
    std = np.std(scores)
    print(f"standard deviation score of training data with LDA: {std}")

    return lda, mean, std, scores

def svm_rbf (X_train, y_train):
    svm = SVC(kernel='rbf', probability = True)

    #cross-validation score
    scores = cross_val_score(svm, X_train, y_train, cv=10)
    mean = np.mean(scores)
    print(f"mean score of training data with SVM :", mean)
    std = np.std(scores)
    print(f"standard deviation score of training data with SVM: {std}")

    return svm, mean, std, scores

def random_forest(X_train, y_train):

        rf = RandomForestClassifier()

        # Create the model 
        n_estim = 200
        rf = RandomForestClassifier(n_estimators=n_estim)

        # cross validation score
        scores = cross_val_score(rf, X_train, y_train, cv=10)
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"mean score of training data with {n_estim}-Random Forest :", mean)
        print(f"standard deviation score of training data with {n_estim}-Random Forest :", std)

        return n_estim, rf, mean, std, scores


def fs_sr(X_train,y_train,X_test,y_test):
    estimator=LogisticRegression(solver='lbfgs', max_iter = 1000,penalty=None, multi_class='multinomial')
    rfecv = RFECV(estimator=estimator, cv=10)
    rfecv.fit(X_train, y_train )

    # Get the selected features from both training and test sets
    features_train = X_train[:, rfecv.get_support()]
    features_test = X_test[:, rfecv.get_support()]
    sr_fs =LogisticRegression(solver='lbfgs',max_iter = 1000, penalty=None, multi_class='multinomial')
    sr_fs.fit(features_train, y_train)
    y_pred_sr_fs = sr_fs.predict(features_test)
    y_pred_sr_fs_proba = sr_fs.predict_proba(features_test)
    

    acc_sr = accuracy_score(y_test, y_pred_sr_fs)
    precision_sr = precision_score(y_test, y_pred_sr_fs, average = 'micro')
    recall_sr = recall_score(y_test, y_pred_sr_fs,average = 'micro')
    f1_sr = f1_score(y_test, y_pred_sr_fs, average = 'micro')
    print(f"accuracy: {acc_sr}, precision: {precision_sr}, recall: {recall_sr}, f1-score:{f1_sr}")
    return y_pred_sr_fs,y_pred_sr_fs_proba

def kmeans(X, y, n_clust, pc):
    kmeans = KMeans(n_clusters=n_clust)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    '''
    fig, axes = plt.subplots(nrows=1, ncols=1)
    #axes[0].scatter(X[:,0], X[:,1], c=y)
    #axes[0].set_title("Original data")
    axes.set_title(f"Kmeans with {pc} pcs and {n_clust} clusters")
    axes.scatter(X[:,0], X[:,1], c=cluster_labels)
    fig.suptitle(f"kmeans with {n_clust} and {pc} PCs")
    
    #plt.savefig(f"figures/kmeans_{n_clust}_{pc}_PCs")
    plt.show()
    '''

    distances = pairwise_distances(X)
    within_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    between_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])

    #Calinski-Harabasz Index
    calinski_score = calinski_harabasz_score(X,cluster_labels)

    #Dunn Index
    min_inter_cluster_distance = np.min([np.min(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])
    max_intra_cluster_distance = np.max([np.max(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    dunn_score = min_inter_cluster_distance / max_intra_cluster_distance

    return silhouette_avg,calinski_score, dunn_score

def gmm(X,y,n_clust,pc):
    clusterer = GaussianMixture(n_components=n_clust)
    clusterer.fit(X)
    cluster_labels = clusterer.predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    '''
    fig, axes = plt.subplots(nrows=1, ncols=1)
    #axes[0].scatter(X[:,0], X[:,1], c=y)
    #axes[0].set_title("Original data")
    axes.set_title(f"Kmeans with {pc} pcs and {n_clust} clusters")
    axes.scatter(X[:,0], X[:,1], c=cluster_labels)
    fig.suptitle(f"GMM with {n_clust} and {pc} PCs")
    
    plt.savefig(f"figures/GMM{n_clust}_{pc}_PCs")
    plt.show()
    '''
    distances = pairwise_distances(X)
    within_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    between_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])

    #Calinski-Harabasz Index
    calinski_score = calinski_harabasz_score(X,cluster_labels)

    #Dunn Index
    min_inter_cluster_distance = np.min([np.min(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])
    max_intra_cluster_distance = np.max([np.max(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    dunn_score = min_inter_cluster_distance / max_intra_cluster_distance
    return silhouette_avg,calinski_score, dunn_score

def dbscan(X,y,n_clust,pc):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    '''
    fig, axes = plt.subplots(nrows=1, ncols=1)
    #axes[0].scatter(X[:,0], X[:,1], c=y)
    #axes[0].set_title("Original data")
    axes.set_title(f"Kmeans with {pc} pcs and {n_clust} clusters")
    axes.scatter(X[:,0], X[:,1], c=cluster_labels)
    fig.suptitle(f"GMM with {n_clust} and {pc} PCs")
    
    plt.savefig(f"figures/GMM{n_clust}_{pc}_PCs")
    plt.show()
    '''
    distances = pairwise_distances(X)
    within_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    between_cluster_scatter = np.sum([np.sum(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])

    #Calinski-Harabasz Index
    calinski_score = calinski_harabasz_score(X,cluster_labels)

    #Dunn Index
    min_inter_cluster_distance = np.min([np.min(distances[cluster_labels == i][:, cluster_labels != i]) for i in range(n_clust)])
    max_intra_cluster_distance = np.max([np.max(distances[cluster_labels == i][:, cluster_labels == i]) for i in range(n_clust)])
    dunn_score = min_inter_cluster_distance / max_intra_cluster_distance
    return silhouette_avg,calinski_score, dunn_score