from curses.ascii import LF
from re import L, M
from load_data import load_data
from q1_functions import drop_missing, class_balance, softmax_regression, naive_bayes,knneigh,linear_discr, svm_rbf,random_forest
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV


X,y,data = load_data()
'''
#Question1a
print("Drop the missing data")
X_missing, y_missing = drop_missing(data)
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
for i in range(1):
    print('run nb', i)
    print("Split into testing and training sets")
    # Classifiers considered : Naive Bayes, Softmax Regression, kNN, Random forest, SVM and QDA
    #split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2, random_state=42,
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
    plt.savefig("figures/confusion_matrix_normalized_seed42")
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
plt.savefig(f"figures/Cross-val_plot")
plt.show()



#Question1b

#Split data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2,  random_state=42,
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()

#ANOVA for feature selection applied on SVN-RBF () -- filter
anova = SelectKBest(score_func=f_classif, k=30)
features = anova.fit_transform(X_train, y_train)

#apply the same selection on the test set
features_test = anova.transform(X_test)

#create the model
svm_feature_sel = SVC(kernel = 'rbf')
#evaluate the model
scores = cross_val_score(svm_feature_sel, features, y_train, cv=10, scoring='accuracy')
std = np.std(scores)
mean = np.mean(scores)
print(f"mean score of training data with SVM-ANOVA :", mean)
std = np.std(scores)
print(f"standard deviation score of training data with SVM-ANOVA: {std}")

#train the model
svm_feature_sel.fit(features, y_train)
y_pred_svm_fs = svm_feature_sel.predict(features_test)
cm_svm_fs = confusion_matrix(y_test, y_pred_svm_fs)

# plot confusion matrices
fig,axes = plt.subplots()
sns.heatmap(cm_svm_fs, annot=True, fmt='g')
sns.heatmap(cm_svm_fs, annot=True, fmt='g')
axes.set_title('SVM with ANOVA normalized_data')
plt.savefig(f"figures/SVM_anova_confusion_mat_normalized_data")
plt.show()

print("LDA L1 reg")
#LDA with L1 regularization - Embedded
lr = LogisticRegressionCV(penalty='l1', solver='saga', cv=10)
lr.fit(X_train, y_train)

coefficients_lr = lr.coef_[0]
index_lr = np.argsort(np.abs(coefficients_lr))[::-1]


selected_features = X_train[:, index_lr[:30]]
print("selected_features.shape",selected_features.shape)

lda_fs = LinearDiscriminantAnalysis()
lda_fs.fit(selected_features, y_train)

#transform the test data using the selected features
X_test_selected = X_test[:, index_lr[:30]]
y_pred_lda_fs = lda_fs.predict(X_test_selected)

#plot confusion matrix
cm_lda_fs = confusion_matrix(y_test, y_pred_lda_fs)
fig,axes = plt.subplots()
sns.heatmap(cm_lda_fs, annot=True, fmt='g')
sns.heatmap(cm_lda_fs, annot=True, fmt='g')
axes.set_title('LDA with lasso regularization normalized_data')
plt.savefig(f"figures/LDA_lasso_confusion_mat_normalized_data")
plt.show()

print("Softmax regression")
#Softmax Regression with Recursive Feature Elimination with Cross-Validation - Wrapper
estimator=LogisticRegression(solver='lbfgs', max_iter = 1000,penalty=None, multi_class='multinomial')
rfecv = RFECV(estimator=estimator, cv=10)
rfecv.fit(X_train, y_train )

# Get the selected features from both training and test sets
features_train = X_train[:, rfecv.get_support()]
features_test = X_test[:, rfecv.get_support()]
sr_fs =LogisticRegression(solver='lbfgs',max_iter = 1000, penalty=None, multi_class='multinomial')
sr_fs.fit(features_train, y_train)
y_pred_sr_fs = sr_fs.predict(features_test)

#plot confusion matrix
cm_sr_fs = confusion_matrix(y_test, y_pred_sr_fs)
fig,axes = plt.subplots()
sns.heatmap(cm_sr_fs, annot=True, fmt='g')
sns.heatmap(cm_sr_fs, annot=True, fmt='g')
axes.set_title('Softmax Regression with Recursive Feature Elimination with Cross-Validation normalized_data')
plt.savefig(f"figures/Softmax_Regression_RFECVconfusion_mat_normalized_data")
plt.show()
'''

#Question 1c 
# Study class balance in the test set
X_missing, y_missing = drop_missing(data)
X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2, 
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()

class_balance(y_train)
