from load_data import load_data
from q1_functions import softmax_regression, naive_bayes, knneigh, linear_discr, svm_rbf, random_forest, fs_sr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

print("imputing data")
# Impute the whole dataset
contaminated = True
X, y,data = load_data()   
imputer = KNNImputer(n_neighbors=7)
X = imputer.fit_transform(X)

print("normalizing data")
# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)


#Question 3a
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.4, random_state=42)

if contaminated == True:
    print("dataset contamination")
    num_samples = int(0.8 * len(y_train))
    #random indices and labels for the mislabeled samples
    mislabel_indices = np.random.choice(len(y_train), num_samples, replace=False)
    incorrect_labels = np.random.choice(np.unique(y_train), num_samples)
    # assign the incorrect labels to the mislabeled samples
    y_train[mislabel_indices] = incorrect_labels

    #same thing fo the validation set with less contamination rate
    num_samples = int(0.1 * len(y_val))
    mislabel_indices = np.random.choice(len(y_val), num_samples, replace=False)
    incorrect_labels = np.random.choice(np.unique(y_val), num_samples)
    y_val[mislabel_indices] = incorrect_labels

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
pca2 = PCA(n_components=30)
X_train_pca2 = pca2.fit_transform(X_train)
X_val_pca2 = pca2.transform(X_val)
X_test_pca2 = pca2.transform(X_test)


print("first train")

model_sr, mean_sr, std_sr, sr_scores = softmax_regression(
    X_train, y_train, None)
model_sr_l2, mean_sr_l2, std_sr_l2, sr_scores_l2 = softmax_regression(
    X_train, y_train, 'l2')
model_nb, mean_nb, std_nb, nb_scores = naive_bayes(X_train, y_train)
model_knn, k, mean_knn, std_knn, knn_scores = knneigh(X_train, y_train)
model_lda, mean_lda, std_lda, lda_scores = linear_discr(X_train, y_train)
model_svm, mean_svm, std_svm, svm_scores = svm_rbf(X_train, y_train)
model_svm_pca, mean_svm_pca, std_svm_pca, svm_scores_pca = svm_rbf(
    X_train_pca, y_train)
model_svm_pca2, mean_svm_pca2, std_svm_pca2, svm_scores_pca2 = svm_rbf(
    X_train_pca2, y_train)
n_estimators, model_rf, mean_rf, std_rf, rf_scores = random_forest(
    X_train, y_train)
#model_fcnn, loss_fcnn, accuracy_fcnn = fcnn(
    #X_train, y_train, X_test, y_test, y)
model_adaboost = adaboost = AdaBoostClassifier(
    n_estimators=50, random_state=42)

print("first fit")
model_sr.fit(X_train, y_train)
model_sr_l2.fit(X_train, y_train)
model_nb.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
model_lda.fit(X_train, y_train)
model_svm.fit(X_train, y_train)
model_svm_pca.fit(X_train_pca, y_train)
model_svm_pca2.fit(X_train_pca2, y_train)
model_rf.fit(X_train, y_train)
#model_fcnn.fit(X_train, y_train, epochs=10, batch_size=32,
#               validation_data=(X_test, y_test))
model_adaboost.fit(X_train, y_train)

print("first predict")
# predict
y_pred_val_sr = model_sr.predict_proba(X_val)
y_pred_val_sr_l2 = model_sr_l2.predict_proba(X_val)
y_pred_sr_fs, y_pred_val_sr_fs_proba = fs_sr(X_train, y_train, X_val, y_val)
y_pred_val_nb = model_nb.predict_proba(X_val)
y_pred_val_knn = model_knn.predict_proba(X_val)
y_pred_val_lda = model_lda.predict_proba(X_val)
y_pred_val_svm = model_svm.predict_proba(X_val)
y_pred_val_svm_pca = model_svm_pca.predict_proba(X_val_pca)
y_pred_val_svm_pca2 = model_svm_pca2.predict_proba(X_val_pca2)
y_pred_val_rf = model_rf.predict_proba(X_val)
#y_pred_val_fcnn = model_fcnn.predict_proba(X_val)
y_pred_val_ada = model_adaboost.predict_proba(X_val)


stacked_X = np.column_stack((y_pred_val_sr, y_pred_val_sr_l2, y_pred_val_sr_fs_proba, y_pred_val_nb, y_pred_val_knn,
                            y_pred_val_lda, y_pred_val_svm, y_pred_val_svm_pca, y_pred_val_svm_pca2, y_pred_val_rf, y_pred_val_ada))

print("ensemble model")
ensemble_model = LogisticRegression(
    solver='lbfgs', max_iter=500, penalty='l2', multi_class='multinomial')


ensemble_model.fit(stacked_X, y_val)


coefs = ensemble_model.coef_[0]
index = np.argsort(np.abs(coefs))[::-1]


#plot the most important features
for i in range(44):
    print(f"{i+1}. feature {index[i]} with importance score {coefs[index[i]]}")

print("second predict")
y_pred_test_sr = model_sr.predict_proba(X_test)
y_pred_test_sr_l2 = model_sr_l2.predict_proba(X_test)
y_pred_sr_fs, y_pred_test_sr_fs_proba = fs_sr(X_train, y_train, X_test, y_test)
y_pred_test_nb = model_nb.predict_proba(X_test)
y_pred_test_knn = model_knn.predict_proba(X_test)
y_pred_test_lda = model_lda.predict_proba(X_test)
y_pred_test_svm = model_svm.predict_proba(X_test)
y_pred_test_svm_pca = model_svm_pca.predict_proba(X_test_pca)
y_pred_test_svm_pca2 = model_svm_pca2.predict_proba(X_test_pca2)
y_pred_test_rf = model_rf.predict_proba(X_test)
#y_pred_test_fcnn = model_fcnn.predict_proba(X_test)
y_pred_test_ada = model_adaboost.predict_proba(X_test)

stack_test = np.column_stack((y_pred_test_sr, y_pred_test_sr_l2, y_pred_test_sr_fs_proba, y_pred_test_nb, y_pred_test_knn,
                             y_pred_test_lda, y_pred_test_svm, y_pred_test_svm_pca, y_pred_test_svm_pca2, y_pred_test_rf, y_pred_test_ada))

print("final predict")
y_pred = ensemble_model.predict(stack_test)

print("calculating scores")
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'micro')
recall = recall_score(y_test, y_pred,average = 'micro')
f1 = f1_score(y_test, y_pred, average = 'micro')
print(f"acc : {acc}, precison :{precision}, recall{recall}, f1{f1}")



print("plotting matrix")
# plot confusion matrices
fig,axes = plt.subplots()
sns.heatmap(cm, annot=True, fmt='g')
axes.set_title('Model Stacking confusion matrix for contaminated dataset')
plt.savefig(f"figures/ModdelStacking_mislabeled_cm_0.8")
plt.show()
'''
#Question 3b

svm = SVC()

param_grid = {
    'C': [0.1, 1, 10],    # the regularization parameter C
    'kernel': ['linear', 'rbf'],  # kernel functions to try
    'gamma': [0.01, 0.1, 1],      # the kernel coefficient gamma
      
}

# Iterate over the parameter grid
classifiers = []
for c in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            print(f"{c},{kernel},{gamma}")
            clf = SVC(C=c, gamma=gamma, kernel=kernel, probability = True)
            # fit the classifier to the data
            clf.fit(X_train, y_train)
            classifiers.append(clf)

y_val_pred =[]
for clf in classifiers:
    y_pred = clf.predict_proba(X_val)
    y_val_pred.append(y_pred)


stack_val = np.column_stack(y_val_pred)
print(stack_val)

print("ensemble model")
ensemble_model = LogisticRegression(
    solver='lbfgs', max_iter=500, penalty='l2', multi_class='multinomial')


ensemble_model.fit(stack_val, y_val)


coefs = ensemble_model.coef_[0]
index = np.argsort(np.abs(coefs))[::-1]


#plot the most important features
for i in range(72):
    print(f"{i+1}. feature {index[i]} with importance score {coefs[index[i]]}")

y_test_pred = []
for clf in classifiers:
    y_pred = clf.predict_proba(X_test)
    y_test_pred.append(y_pred)

stack_test = np.column_stack(y_test_pred)


print("final predict")
y_pred = ensemble_model.predict(stack_test)

print("calculating scores")
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'micro')
recall = recall_score(y_test, y_pred,average = 'micro')
f1 = f1_score(y_test, y_pred, average = 'micro')
print(f"acc : {acc}, precison :{precision}, recall{recall}, f1{f1}")



print("plotting matrix")
# plot confusion matrices
fig,axes = plt.subplots()
sns.heatmap(cm, annot=True, fmt='g')
axes.set_title('Model Stacking confusion matrix With SVM')
plt.savefig(f"figures/SVM_model_stacking_cm")
plt.show()
'''