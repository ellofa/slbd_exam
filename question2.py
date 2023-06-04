from load_data import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from q1_functions import  softmax_regression, naive_bayes,knneigh,linear_discr, svm_rbf,random_forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge



X,y,data = load_data()
'''
#question 2a
# Identify missing values
data_X = data.loc[:,:] 
data_y = data.loc[:,'0'] 

missing_data_per_obs = data_X.isna().sum(axis=1)
#print(missing_data_per_obs)
missing_data_per_feature = data_X.isna().sum(axis=0)
missing_data_distribution = data.isnull().mean(axis=1)
print(missing_data_distribution)

#plot the distribution of missing data per feature
plt.hist(missing_data_per_obs, bins=range(1, np.max(missing_data_per_obs) + 2), edgecolor='black')
plt.xlabel('Number of Missing Features')
plt.ylabel('Frequency')
plt.title('Distribution of missing data per observation')
#plt.savefig("figures/missing_data_per_obs")
plt.show()


#plot the distribution of missing data per observation
plt.hist(missing_data_per_feature, bins=range(300, np.max(missing_data_per_feature) + 2), edgecolor='black')
plt.xlabel('Number of Missing Features')
plt.ylabel('Frequency')
plt.title('Distribution of missing data per feature')
#plt.savefig("figures/missing_data_per_feature")
plt.show()

data_X.dropna(inplace=True)
print("shape clean", data_X.shape)


for i in range(data_X.shape[0]):
    # missingness distribution for feature 
    bernoulli_pick = np.random.binomial(1, 0.6, 1)
    #print(bernoulli_pick)
    obs_missing_data_ratio = missing_data_distribution.iloc[i]
    #print(obs_missing_data_ratio)
    pick = np.random.rand(1)
    if (pick < obs_missing_data_ratio) and (bernoulli_pick[0] == 1):
        data_X.iloc[i, :] = np.nan

data_X.dropna(inplace=True)
print("shape after mimicing missing vals", data_X.shape)

X_missing = data_X.loc[:, data_X.columns != '0'].to_numpy()
y_missing = np.ravel(data_X.loc[:,'0'].to_numpy())

print("Normalize data")
scaler = Normalizer().fit(X_missing)
X_missing = scaler.transform(X_missing)


m = [0,0,0,0,0,0]
s = [0,0,0,0,0,0]
sr_scores_sum = np.array([0 for i in range(10)])
nb_scores_sum = np.array([0 for i in range(10)])
knn_scores_sum = np.array([0 for i in range(10)])
lda_scores_sum = np.array([0 for i in range(10)])
svm_scores_sum = np.array([0 for i in range(10)])
rf_scores_sum = np.array([0 for i in range(10)])
for i in range(10):
    print('run nb', i)
    print("Split into testing and training sets")
    # Classifiers considered : Naive Bayes, Softmax Regression, kNN, Random forest, SVM and QDA
    #split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2,
                                                                shuffle=True)
    
    #print("Dimension reduction for the training set")
    #pca = PCA(n_components=10) 
    #X_train = pca.fit_transform(X_train)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model_sr, mean_sr, std_sr, sr_scores = softmax_regression(X_train,y_train, None)
    model_nb, mean_nb, std_nb, nb_scores = naive_bayes(X_train,y_train)
    model_knn, k,mean_knn, std_knn, knn_scores = knneigh(X_train,y_train)
    model_lda, mean_lda, std_lda, lda_scores = linear_discr(X_train,y_train)
    model_svm, mean_svm, std_svm, svm_scores = svm_rbf(X_train,y_train)
    n_estimators, model_rf, mean_rf, std_rf, rf_scores = random_forest(X_train,y_train)

    # Create a list of mean cross val scores and stds for each method
    means = [mean_sr, mean_nb,mean_knn, mean_lda, mean_svm, mean_rf]
    stds = [std_sr, std_nb, std_knn, std_lda, std_svm, std_rf]

    m[0]+=mean_sr
    m[1]+=mean_nb
    m[2]+=mean_knn
    m[3]+=mean_lda
    m[4]+=mean_svm
    m[5]+=mean_rf

    s[0]+= std_sr
    s[1]+= std_nb
    s[2]+= std_knn
    s[3]+= std_lda
    s[4]+= std_svm
    s[5]+= std_rf

    print("Computing the scores")
    sr_scores_sum = sr_scores_sum + np.array(sr_scores)
    nb_scores_sum = nb_scores_sum + np.array(nb_scores)
    knn_scores_sum = knn_scores_sum + np.array(knn_scores)
    lda_scores_sum = lda_scores_sum + np.array(lda_scores)
    svm_scores_sum = svm_scores_sum + np.array(svm_scores)
    rf_scores_sum = rf_scores_sum + np.array(rf_scores)
    print(sr_scores)
    print(sr_scores_sum)




    #fit models
    model_sr.fit(X_train,y_train)
    model_nb.fit(X_train,y_train)
    model_knn.fit(X_train,y_train)
    model_lda.fit(X_train,y_train)
    model_svm.fit(X_train,y_train)
    model_rf.fit(X_train,y_train)

    #predict
    y_pred_sr = model_sr.predict(X_test)
    y_pred_nb = model_nb.predict(X_test)
    y_pred_knn = model_knn.predict(X_test)
    y_pred_lda = model_lda.predict(X_test)
    y_pred_svm = model_svm.predict(X_test)
    y_pred_rf = model_rf.predict(X_test)

    # calculate confusion matrix for each model
    cm_sr = confusion_matrix(y_test, y_pred_sr)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_lda = confusion_matrix(y_test, y_pred_lda)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # calculate accuracy
    acc_sr = accuracy_score(y_test, y_pred_sr)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # calculate precision
    precision_sr = precision_score(y_test, y_pred_sr, average = 'micro')
    precision_nb = precision_score(y_test, y_pred_nb, average = 'micro')
    precision_knn = precision_score(y_test, y_pred_knn, average = 'micro')
    precision_lda = precision_score(y_test, y_pred_lda, average = 'micro')
    precision_svm = precision_score(y_test, y_pred_nb, average = 'micro')
    precision_rf = precision_score(y_test, y_pred_rf, average = 'micro')

    # calculate recall
    recall_sr = recall_score(y_test, y_pred_sr,average = 'micro')
    recall_nb = recall_score(y_test, y_pred_nb, average = 'micro')
    recall_knn = recall_score(y_test, y_pred_knn, average = 'micro')
    recall_lda = recall_score(y_test, y_pred_lda, average = 'micro')
    recall_svm = recall_score(y_test, y_pred_svm, average = 'micro')
    recall_rf = recall_score(y_test, y_pred_rf, average = 'micro')

    # calculate F1-score
    f1_sr = f1_score(y_test, y_pred_sr, average = 'micro')
    f1_nb = f1_score(y_test, y_pred_nb, average = 'micro')
    f1_knn = f1_score(y_test, y_pred_knn, average = 'micro')
    f1_lda = f1_score(y_test, y_pred_lda, average = 'micro')
    f1_svm = f1_score(y_test, y_pred_svm, average = 'micro')
    f1_rf = f1_score(y_test, y_pred_rf, average = 'micro')


    # print the results
    print('Accuracy: NB:', acc_nb,  'SR: ', acc_sr, 'KNN:', acc_lda,'SVM', acc_svm,'RF: ', acc_rf)
    print('Precision: NB:',precision_nb, 'SR: ', precision_sr,'KNN:', precision_knn,'LDA:', precision_lda,'SVM:', precision_svm, 'RF: ', precision_rf)
    print('Recall: NB:',recall_nb, 'SR: ', recall_sr ,  'KNN:', recall_knn,'LDA:', recall_lda,'SVM:', recall_svm,'RF: ', recall_rf)
    print('F1-score: NB:', f1_nb, 'SR: ', f1_sr, 'KNN:', f1_knn,'LDA:',f1_lda,'SVM:', f1_svm, 'RF: ', f1_rf)
    
    # plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(12, 4))

    sns.heatmap(cm_sr, annot=True, fmt='g', ax=axes[0,0])
    sns.heatmap(cm_nb, annot=True, fmt='g', ax=axes[0,1])
    sns.heatmap(cm_knn, annot=True, fmt='g', ax=axes[0,2])
    sns.heatmap(cm_lda, annot=True, fmt='g', ax=axes[1,0])
    sns.heatmap(cm_svm, annot=True, fmt='g', ax=axes[1,1])
    sns.heatmap(cm_rf, annot=True, fmt='g', ax=axes[1,2])

    axes[0,0].set_title('Softmax Regression')
    axes[0,1].set_title('Naive-Bayes')
    axes[0,2].set_title(f'{k}-Nearest Neighbors')
    axes[1,0].set_title('Linear Discriminant Analysis')
    axes[1,1].set_title('SVM')
    axes[1,2].set_title('Random Forest')


    plt.tight_layout()
    plt.savefig("figures/confusion_matrix_normalized_q2")
    plt.show()
    fig.canvas.flush_events()
    

for i in range(len(m)):
    m[i]/=10
    s[i]/=10
for i in range(10):
    sr_scores_sum[i] /=10
    nb_scores_sum[i] /=10
    knn_scores_sum[i] /=10
    lda_scores_sum[i] /=10
    svm_scores_sum[i] /=10
    rf_scores_sum[i] /=10

# Create a list of method names
methods = ['Softmax Regression' , 'Naive Bayes', f'{k}-NN', 'Linear Discriminant Analysis', 'SVM_RBF', 'Random Forest']
# Plot the box plots
fig, ax = plt.subplots()
#boxprops = dict(linestyle='-', linewidth=1.5, color='k', markerfacecolor='None', markeredgecolor='black')
ax.boxplot([sr_scores_sum, nb_scores_sum, knn_scores_sum, lda_scores_sum, svm_scores_sum, rf_scores_sum], labels=methods, showmeans=True)

# Add horizontal lines for the mean cross val score and std for each method
for i in range(len(methods)):
    #ax.axhline(y=means[i], color='r', linestyle='--')
    ax.annotate('Mean = {:.3f}\nStd = {:.3f}'.format(m[i], s[i]), xy=(i+1, m[i]), xytext=(20, 5),
                textcoords='offset points', ha='left', va='bottom', fontsize=10, color='grey')

# Set plot title and axis labels
ax.set_title('Cross Validation Scores')
ax.set_xlabel('Classification Method')
ax.set_ylabel('Cross Validation Score')

#Save figure
plt.savefig(f"figures/Cross-val_plot_q2_10runs")
plt.show()
'''

#question 2b
missing_indices = np.isnan(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()
'''
#SIMPLE IMPUTATATION
imputer = SimpleImputer(strategy='mean')
X_train_simple_imputor = imputer.fit_transform(X_train)
X_test_simple_imputor = imputer.transform(X_test)

# KNN IMPUTER
imputer = KNNImputer(n_neighbors=7) 
X_train_knn_imputor = imputer.fit_transform(X_train)
X_test_knn_imputor = imputer.transform(X_test)
'''
print('iterative imputor')
# ITERATIVE IMPUTER
imputer = IterativeImputer(estimator=BayesianRidge(),initial_strategy = 'mean',n_nearest_features = 10,random_state=42)  
X_train_iterative_imputor = imputer.fit_transform(X_train)
X_test_iterative_imputor = imputer.transform(X_test)

# change X_train to change method of imputation
X_train_imputed = X_train_iterative_imputor
X_test_imputed = X_test_iterative_imputor

#Fitting the models  and evaluating the means and stds
means = []
stds = []
model_sr, mean_sr, std_sr, sr_scores = softmax_regression(X_train_imputed,y_train, None)
means.append(mean_sr)
stds.append(std_sr)
model_nb, mean_nb, std_nb, nb_scores = naive_bayes(X_train_imputed,y_train)
means.append(mean_nb)
stds.append(std_nb)
model_knn, k,mean_knn, std_knn, knn_scores = knneigh(X_train_imputed,y_train)
means.append(mean_knn)
stds.append(std_knn)
model_lda, mean_lda, std_lda, lda_scores = linear_discr(X_train_imputed,y_train)
means.append(mean_lda)
stds.append(std_lda)
model_svm, mean_svm, std_svm, svm_scores = svm_rbf(X_train_imputed,y_train)
means.append(mean_svm)
stds.append(std_svm)
n_estimators, model_rf, mean_rf, std_rf, rf_scores = random_forest(X_train_imputed,y_train)
means.append(mean_rf)
stds.append(std_rf)

# Create a list of method names
methods = ['Softmax Regression' , 'Naive Bayes', f'{k}-NN', 'Linear Discriminant Analysis', 'SVM_RBF', 'Random Forest']
# Plot the box plots
fig, ax = plt.subplots()
#boxprops = dict(linestyle='-', linewidth=1.5, color='k', markerfacecolor='None', markeredgecolor='black')
ax.boxplot([sr_scores, nb_scores, knn_scores, lda_scores, svm_scores, rf_scores], labels=methods, showmeans=True)

# Add horizontal lines for the mean cross val score and std for each method
for i in range(len(methods)):
    #ax.axhline(y=means[i], color='r', linestyle='--')
    ax.annotate('Mean = {:.3f}\nStd = {:.3f}'.format(means[i], stds[i]), xy=(i+1, means[i]), xytext=(20, 5),
                textcoords='offset points', ha='left', va='bottom', fontsize=10, color='grey')

# Set plot title and axis labels
ax.set_title('Cross Validation Scores Using the Iterative Imputor')
ax.set_xlabel('Classification Method')
ax.set_ylabel('Cross Validation Score')

#Save figure
plt.savefig(f"figures/Cross-val_plot_iterative_imputor")
plt.show()

#fit models
model_sr.fit(X_train_imputed,y_train)
model_nb.fit(X_train_imputed,y_train)
model_knn.fit(X_train_imputed,y_train)
model_lda.fit(X_train_imputed,y_train)
model_svm.fit(X_train_imputed,y_train)
model_rf.fit(X_train_imputed,y_train)

#predict
y_pred_sr = model_sr.predict(X_test_imputed)
y_pred_nb = model_nb.predict(X_test_imputed)
y_pred_knn = model_knn.predict(X_test_imputed)
y_pred_lda = model_lda.predict(X_test_imputed)
y_pred_svm = model_svm.predict(X_test_imputed)
y_pred_rf = model_rf.predict(X_test_imputed)

# calculate confusion matrix for each model
cm_sr = confusion_matrix(y_test, y_pred_sr)
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_rf = confusion_matrix(y_test, y_pred_rf)

# calculate accuracy
acc_sr = accuracy_score(y_test, y_pred_sr)
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_lda = accuracy_score(y_test, y_pred_lda)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_rf = accuracy_score(y_test, y_pred_rf)

# calculate precision
precision_sr = precision_score(y_test, y_pred_sr, average = 'micro')
precision_nb = precision_score(y_test, y_pred_nb, average = 'micro')
precision_knn = precision_score(y_test, y_pred_knn, average = 'micro')
precision_lda = precision_score(y_test, y_pred_lda, average = 'micro')
precision_svm = precision_score(y_test, y_pred_nb, average = 'micro')
precision_rf = precision_score(y_test, y_pred_rf, average = 'micro')

# calculate recall
recall_sr = recall_score(y_test, y_pred_sr,average = 'micro')
recall_nb = recall_score(y_test, y_pred_nb, average = 'micro')
recall_knn = recall_score(y_test, y_pred_knn, average = 'micro')
recall_lda = recall_score(y_test, y_pred_lda, average = 'micro')
recall_svm = recall_score(y_test, y_pred_svm, average = 'micro')
recall_rf = recall_score(y_test, y_pred_rf, average = 'micro')

# calculate F1-score
f1_sr = f1_score(y_test, y_pred_sr, average = 'micro')
f1_nb = f1_score(y_test, y_pred_nb, average = 'micro')
f1_knn = f1_score(y_test, y_pred_knn, average = 'micro')
f1_lda = f1_score(y_test, y_pred_lda, average = 'micro')
f1_svm = f1_score(y_test, y_pred_svm, average = 'micro')
f1_rf = f1_score(y_test, y_pred_rf, average = 'micro')


# print the results
print('Accuracy: NB:', acc_nb,  'SR: ', acc_sr, 'KNN:', acc_lda,'SVM', acc_svm,'RF: ', acc_rf)
print('Precision: NB:',precision_nb, 'SR: ', precision_sr,'KNN:', precision_knn,'LDA:', precision_lda,'SVM:', precision_svm, 'RF: ', precision_rf)
print('Recall: NB:',recall_nb, 'SR: ', recall_sr ,  'KNN:', recall_knn,'LDA:', recall_lda,'SVM:', recall_svm,'RF: ', recall_rf)
print('F1-score: NB:', f1_nb, 'SR: ', f1_sr, 'KNN:', f1_knn,'LDA:',f1_lda,'SVM:', f1_svm, 'RF: ', f1_rf)

# plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(12, 4))

sns.heatmap(cm_sr, annot=True, fmt='g', ax=axes[0,0])
sns.heatmap(cm_nb, annot=True, fmt='g', ax=axes[0,1])
sns.heatmap(cm_knn, annot=True, fmt='g', ax=axes[0,2])
sns.heatmap(cm_lda, annot=True, fmt='g', ax=axes[1,0])
sns.heatmap(cm_svm, annot=True, fmt='g', ax=axes[1,1])
sns.heatmap(cm_rf, annot=True, fmt='g', ax=axes[1,2])

axes[0,0].set_title('Softmax Regression')
axes[0,1].set_title('Naive-Bayes')
axes[0,2].set_title(f'{k}-Nearest Neighbors')
axes[1,0].set_title('Linear Discriminant Analysis')
axes[1,1].set_title('SVM')
axes[1,2].set_title('Random Forest')


plt.tight_layout()
plt.savefig("figures/confusion_matrix_iterative_imputor")
plt.show()
fig.canvas.flush_events()