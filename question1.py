from load_data import load_data
from q1_functions import drop_missing, class_balance, softmax_regression, naive_bayes,knneigh,linear_discr, svm_rbf,random_forest,fs_sr,kmeans,gmm,dbscan 
import numpy as np
import pandas as pd
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFECV
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import KMeans


X,y,data = load_data()
X_missing, y_missing,data = drop_missing(data)
scaler = Normalizer().fit(X_missing)
X_missing = scaler.transform(X_missing)
'''
#Question1a
print("Drop the missing data")
X_missing, y_missing,data = drop_missing(data)
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


'''
#Question1b

#Split data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_missing, y_missing, test_size=0.2,  random_state=42,
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()
y_missing = y_missing.ravel()
'''
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

# plot confusion matrice
fig,axes = plt.subplots()
sns.heatmap(cm_svm_fs, annot=True, fmt='g')
axes.set_title('SVM with ANOVA normalized_data')
#plt.savefig(f"figures/SVM_anova_confusion_mat_normalized_data")
plt.show()

print("LDA L1 reg")

#LDA with L1 regularization - Embedded
lr = LogisticRegressionCV(penalty='l1', solver='saga', cv=10)
lr.fit(X_train, y_train)

coefficients_lr = lr.coef_[0]
index_lr = np.argsort(np.abs(coefficients_lr))[::-1]
selected_features = X_train[:, index_lr[:100]]

#plot the most important features
index_100 = []
importance_100 = []
for i in range(100):
    print(f"{i+1}. feature {index_lr[i]} with importance score {coefficients_lr[index_lr[i]]}")
    index_100.append(index_lr[i])
    importance_100.append(abs(coefficients_lr[index_lr[i]]))

# plot feature importance
fig, ax = plt.subplots()
ax.bar(np.arange(100), importance_100, align='center')
ax.set_xticks(np.arange(100))
ax.set_xticklabels(index_100)
ax.set_ylabel('Importance')
ax.set_title('Top 100 Important Features with Lasso Regularization')
plt.savefig(f"figures/100f_Lasso")
plt.show()



lda_fs = LinearDiscriminantAnalysis()
lda_fs.fit(selected_features, y_train)

#transform the test data using the selected features
X_test_selected = X_test[:, index_lr[:100]]
y_pred_lda_fs = lda_fs.predict(X_test_selected)

#plot confusion matrix
cm_lda_fs = confusion_matrix(y_test, y_pred_lda_fs)
fig,axes = plt.subplots()
sns.heatmap(cm_lda_fs, annot=True, fmt='g')
axes.set_title('LDA with lasso regularization normalized_data')
#plt.savefig(f"figures/LDA_lasso_confusion_mat_normalized_data")
plt.show()

print("Softmax regression")
#Softmax Regression with Recursive Feature Elimination with Cross-Validation - Wrapper
estimator=LogisticRegression(solver='lbfgs', max_iter = 1000,penalty=None, multi_class='multinomial')
rfecv = RFECV(estimator=estimator, cv=10)
rfecv.fit(X_train, y_train )
#for i in range(1000) :
#    print ("feature ",i, " is ranked : ", rfecv.ranking_[i])
# Plot the feature rankings
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Feature rank")
plt.plot(range(1, len(rfecv.ranking_) + 1), rfecv.ranking_, marker='o')
plt.title("RFECV - Feature Rankings")
plt.savefig("figures/RFECV_Feature Rankings")
plt.show()

# Get the selected features from both training and test sets
features_train = X_train[:, rfecv.get_support()]
features_test = X_test[:, rfecv.get_support()]
sr_fs =LogisticRegression(solver='lbfgs',max_iter = 1000, penalty=None, multi_class='multinomial')
sr_fs.fit(features_train, y_train)
y_pred_sr_fs = sr_fs.predict(features_test)

acc_sr = accuracy_score(y_test, y_pred_sr_fs)
precision_sr = precision_score(y_test, y_pred_sr_fs, average = 'micro')
recall_sr = recall_score(y_test, y_pred_sr_fs,average = 'micro')
f1_sr = f1_score(y_test, y_pred_sr_fs, average = 'micro')
print(f"accuracy: {acc_sr}, precision: {precision_sr}, recall: {recall_sr}, f1-score:{f1_sr}")

#plot confusion matrix
cm_sr_fs = confusion_matrix(y_test, y_pred_sr_fs)
fig,axes = plt.subplots()
sns.heatmap(cm_sr_fs, annot=True, fmt='g')
axes.set_title('Softmax Regression with Recursive Feature Elimination with Cross-Validation normalized_data')
#plt.savefig(f"figures/Softmax_Regression_RFECVconfusion_mat_normalized_data")
plt.show()


#Question 1c 
#merging classes
X_missing_, y_missing_,data_ = drop_missing(data)

data_['0'] = data_['0'].replace(1, 2)
X_missing_ = data_.loc[:, data_.columns != '0'].to_numpy()
y_missing_ = np.ravel(data_.loc[:,'0'].to_numpy())
#class_balance(y_missing)

X_train, X_test, y_train, y_test = train_test_split(X_missing_, y_missing_, test_size=0.2, 
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()


# Compute pairwise distances using Euclidean distance metric
distances = pdist(X_train, metric='euclidean')
distance_matrix = squareform(distances)
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_indices = np.where(y_train == cls)[0]
    class_distances = distance_matrix[class_indices, :][:, class_indices]
    mean_distance = np.mean(class_distances)
    print(f"Class: {cls}, Mean Distance: {mean_distance}")

model_sr, mean_sr, std_sr, sr_scores = softmax_regression(X_train,y_train, None)
model_sr.fit(X_train,y_train)
y_pred_merge = model_sr.predict(X_test)

cm_sr_merge = confusion_matrix(y_test, y_pred_merge)
acc_sr = accuracy_score(y_test, y_pred_merge)
precision_sr = precision_score(y_test, y_pred_merge, average = 'micro')
recall_sr = recall_score(y_test, y_pred_merge,average = 'micro')
f1_sr = f1_score(y_test, y_pred_merge, average = 'micro')
print(f"accuracy: {acc_sr}, precision: {precision_sr}, recall: {recall_sr}, f1-score:{f1_sr}")

fig,axes = plt.subplots()
sns.heatmap(cm_sr_merge, annot=True, fmt='g')
sns.heatmap(cm_sr_merge, annot=True, fmt='g')
axes.set_title('Merging class 1 and 2')
#plt.savefig(f"figures/merge1_2")
plt.show()

print("after applying feature selection")
y_pred_sr_fs,y_pred_sr_fs_proba = fs_sr(X_train,y_train,X_test,y_test)


#splitting classes
#first approach (failed)

# Calculate the correlation within each class
min = 2
low_corr_cls = -1
for cls in unique_classes:
    class_features = X_missing[y_missing == cls]
    correlation_matrix = np.corrcoef(class_features, rowvar=False)
    average_correlation = np.mean(correlation_matrix)
    
    #average correlation for the class
    print("Class:",cls)
    print("Average Correlation:", average_correlation)
    if average_correlation < min :
        min = average_correlation
        low_corr_cls = cls
print(low_corr_cls)
'''
'''
print("merge then split")
#second approach
#dataframe containing only the class to divide into 2
y_missing_ = y_missing_.ravel()
to_divide = X_missing_[np.logical_or(y_missing_ == 1, y_missing_ == 2)]
kmeans = KMeans(n_clusters=3)
cluster_labels = kmeans.fit_predict(to_divide)
y_clust = cluster_labels
 
label_mapping = {
    0: 5,
    1: 6,
    2: 7
}

# Update the labels in the numpy array
for i, label in enumerate(cluster_labels):
    y_clust[i] = label_mapping[label]

y_clust = y_clust.ravel()

data_new = pd.DataFrame(np.concatenate([np.expand_dims(y_clust, axis=1),to_divide], axis=1))
print(data_new)
print(data_new.shape)

#drop the label 2 and 1
data_ = data
data_ = data_.drop(data_[data_['0'] == 1].index)
data_ = data_.drop(data_[data_['0'] == 2].index)
print(data.shape)

#add the new labels
data_new.columns = data.columns
data_clust = pd.concat([data_, data_new], ignore_index=True)
print(data_clust.shape)
print(data_clust)

#classify with softmax regression 
X_clust = data_clust.loc[:, data_clust.columns != '0'].to_numpy()
y_clust = np.ravel(data_clust.loc[:,'0'].to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X_clust,y_clust, test_size=0.2, 
                                                                shuffle=True)
y_train = y_train.ravel()
y_test = y_test.ravel()

#class_balance(y_train)
#print()

model_sr, mean_sr, std_sr, sr_scores = softmax_regression(X_train,y_train, None)
model_sr.fit(X_train,y_train)
y_pred_merge = model_sr.predict(X_test)

cm_sr_merge = confusion_matrix(y_test, y_pred_merge)
acc_sr = accuracy_score(y_test, y_pred_merge)
precision_sr = precision_score(y_test, y_pred_merge, average = 'micro')
recall_sr = recall_score(y_test, y_pred_merge,average = 'micro')
f1_sr = f1_score(y_test, y_pred_merge, average = 'micro')
print(f"accuracy: {acc_sr}, precision: {precision_sr}, recall: {recall_sr}, f1-score:{f1_sr}")

fig,axes = plt.subplots()
sns.heatmap(cm_sr_merge, annot=True, fmt='g')
sns.heatmap(cm_sr_merge, annot=True, fmt='g')
axes.set_title('Splitting class 1 and 2 into 3 clusters')
#plt.savefig(f"figures/split_1&2_into_3")
plt.show()
print("after applying feature selection")
fs_sr(X_train,y_train,X_test,y_test)

#Question1d
X_class3 = X_missing[y_missing == 3]
X_class1 = X_missing[y_missing == 1]
variance_1 = []
variance_3 = []
variance_total = []
for feature in importance_100:
    feature_variance1 = np.var(X_class1[:, int(feature)])
    variance_1.append(feature_variance1)
    feature_variance3 = np.var(X_class3[:, int(feature)])
    variance_3.append(feature_variance3)
    feature_variance_total = np.var(X_missing[:, int(feature)])
    variance_total.append(feature_variance_total)
# plot feature importance
fig, ax = plt.subplots()

default_x_ticks = np.arange(100)
plt.plot(default_x_ticks, variance_3)
plt.xticks(default_x_ticks, importance_100)

ax.set_ylabel('Importance')
ax.set_title('Variance of top 100 Important Features for Class 3')
#plt.savefig(f"figures/100f_Lasso")
plt.show()
'''
# Compare class 1 versus class 2
features = X_missing[np.logical_or(y_missing == 1, y_missing == 2)]
class_labels = y_missing[np.logical_or(y_missing == 1, y_missing == 2)].reshape(871,1)

#class_of_interest = 3

p_value_threshold = 0.05


correlation_coeffs = []
adjusted_p_values = []

for feature in features.T:
    #binary_labels = np.where(class_labels == class_of_interest, 1, 0)
    
    # calculate the correlation coefficient and p-value
    correlation_coeff, p_value = stats.pearsonr(feature, class_labels)
    correlation_coeffs.append(correlation_coeff)
    adjusted_p_values.append(p_value)

# asjust p-values using the Benjamini-Hochberg procedure
adjusted_p_values = multipletests(adjusted_p_values, method='fdr_bh')[1]



# sort the feature results based on the adjusted p-value (ascending) and correlation coefficient
feature_results = [(i, correlation_coeffs[i], adjusted_p_values[i]) for i in range(len(features.T))]
sorted_features = sorted(feature_results, key=lambda x: (x[2], -abs(x[1])))

for feature in sorted_features:
    feature_index, correlation_coeff, adjusted_p_value = feature
    print("Feature", feature_index+1)
    print("Pearson correlation coefficient:", correlation_coeff)
    print("Adjusted p-value:", adjusted_p_value)
    print()


#Queston 1e

print("Question 1e")
X_missing, y_missing,data = drop_missing(data)
y_missing = y_missing.ravel()
#print(data.shape)

# PCA on the data : 
print("Dimension reduction for data set")
X_missing_pca = data.loc[:, data.columns != '0'].to_numpy()
pca = PCA(n_components=10) 
X_missing_pca = pca.fit_transform(X_missing)


total_silhouette_avg_km = np.array([0,0,0])
total_c_km = np.array([0,0,0])
total_d_km = np.array([0,0,0])

total_silhouette_avg_gmm = np.array([0,0,0])
total_c_gmm = np.array([0,0,0])
total_d_gmm = np.array([0,0,0])
for j in range(10):
    print("run",j)
    clusters =[3,4,5]
    silhouette_km = []
    dunn_km = []
    calinski_km = []

    silhouette_gmm = []
    dunn_gmm = []
    calinski_gmm = []
    for i in clusters:
        #print(i)
        silhouette_avg_kmeans,c_km,d_km = kmeans(X_missing_pca,y_missing ,i,30)
        silhouette_km.append(silhouette_avg_kmeans)
        calinski_km.append(c_km)
        dunn_km.append(d_km)

        silhouette_avg_gmm,c_gmm,d_gmm = gmm(X_missing_pca,y_missing,i,30)
        silhouette_gmm.append(silhouette_avg_gmm)
        calinski_gmm.append(c_gmm)
        dunn_gmm.append(d_gmm)
    silhouette_avg_db,c_db,d_db = dbscan(X_missing_pca,y_missing,i,100)

    total_silhouette_avg_km = total_silhouette_avg_km + np.array(silhouette_km)
    total_c_km = total_c_km + np.array(calinski_km)
    total_d_km = total_d_km + np.array(dunn_km)

    total_silhouette_avg_gmm = total_silhouette_avg_gmm + np.array(silhouette_gmm)
    total_c_gmm = total_c_gmm + np.array(calinski_gmm)
    total_d_gmm = total_d_gmm + np.array(dunn_gmm)


for i in range(len(total_silhouette_avg_km)):
    total_silhouette_avg_km[i]/=10
    total_c_km[i]/=10
    total_d_km[i]/=10

    total_silhouette_avg_gmm[i]/=10
    total_c_gmm[i]/=10
    total_d_gmm[i]/=10
fig = plt.figure("fig1")
plt.plot(clusters, total_silhouette_avg_km, marker='o', color='green',label = 'kmeans')
plt.plot(clusters, total_silhouette_avg_gmm, marker='o',label = 'gmm')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.savefig(f"figures/silhouette_30pcs")
plt.show()

fig = plt.figure("fig2")
plt.plot(clusters, total_c_km, marker='o', color='green')
plt.plot(clusters, total_c_gmm, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski Index')
plt.title('Calinski Index for Different Numbers of Clusters')
plt.savefig(f"figures/calinski_30pcs")
plt.show()

fig = plt.figure("fig3")
plt.plot(clusters, total_d_km, marker='o', color='green')
plt.plot(clusters, total_d_gmm, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Dunn Index')
plt.title('Dunn Index for Different Numbers of Clusters')
plt.savefig(f"figures/dunn_30pcs")
plt.show()

#Feature Clustering
X_features = X_missing.T
X_missing_pca = data.loc[:, data.columns != '0'].to_numpy()
X_features = X_missing_pca.T
print("shape :",X_features.shape)
pca = PCA(n_components=100) 
X_features_pca = pca.fit_transform(X_features)

clusters =[4,5,6]
silhouette_km = []
dunn_km = []
calinski_km = []

silhouette_gmm = []
dunn_gmm = []
calinski_gmm = []
for i in clusters:
    silhouette_avg_kmeans,c_km,d_km = kmeans(X_features_pca,y_missing ,i,100)
    silhouette_avg_gmm,c_gmm,d_gmm = gmm(X_features_pca,y_missing,i,100)

    silhouette_km.append(silhouette_avg_kmeans)
    calinski_km.append(c_km)
    dunn_km.append(d_km)

   
    silhouette_gmm.append(silhouette_avg_gmm)
    calinski_gmm.append(c_gmm)
    dunn_gmm.append(d_gmm)


fig = plt.figure("fig1")
plt.plot(clusters, silhouette_km, marker='o', color='yellow', label ='kmeans')
plt.plot(clusters, silhouette_gmm, marker='o', color='green', label ='gmm')
plt.legend(loc='upper left')
plt.xlabel('Number of Clusters')
plt.ylabel('Kmeans Scores')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.savefig(f"figures/silhouette_scores_feature_clustering")
plt.show()

fig = plt.figure("fig2")
plt.plot(clusters, calinski_km, marker='o',  color = 'yellow',label = 'kmeans')
plt.plot(clusters, calinski_gmm, marker='o',  color = 'green',label = 'gmm')
plt.legend(loc='upper right')
plt.xlabel('Number of Clusters')
plt.ylabel('GMM Scores')
plt.title('Calinski Index for Different Numbers of Clusters using GMM')
plt.savefig(f"figures/calinski_scores_feature_clustering")
plt.show()

fig = plt.figure("fig3")
plt.plot(clusters, dunn_km, marker='o', color = 'yellow', label = 'kmeans')
plt.plot(clusters, dunn_km, marker='o', color = 'green', label = 'gmm')
plt.legend(loc='upper left')
plt.xlabel('Number of Clusters')
plt.ylabel('Kmeans Scores')
plt.title('Dunn Index for Different Numbers of Clusters')
plt.savefig(f"figures/dunn_scores_feature_clustering")
plt.show()

