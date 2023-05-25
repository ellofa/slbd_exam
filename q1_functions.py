from load_data import load_data
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def drop_missing(data):
    data[data == -1000] = np.nan
    data.dropna(inplace=True)
    n = data.shape[0]
    X = data.loc[:, data.columns != '0'].to_numpy()
    y = data.loc[:,'0'].to_numpy().reshape(n,1)
    return X,y

def class_balance(y):
    print("nb of samples belonging to class 1", np.sum(y == 1))
    print("nb of samples belonging to class 2", np.sum(y == 2))
    print("nb of samples belonging to class 3", np.sum(y == 3))
    print("nb of samples belonging to class 4", np.sum(y == 4))

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
    svm = SVC(kernel='rbf')

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






