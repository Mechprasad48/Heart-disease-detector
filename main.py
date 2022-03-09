'''Code by Prasad Sawantdesai'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


'''First Reading the data file'''
data = pd.read_csv('heart.csv')
'''Checkout the feature of data-set'''
d = data.head()
print(d)
'''Check out no. of rows and column'''
p = data.shape
print(p)
'''feature engineering'''
'''To check-out if any value in data-set is filled with null'''
o = data.isnull().sum()
print(o)
'''to check-out outlier in data-set'''
'''outliers- what is outlier with example?
a value that "lies outside" (is much smaller or larger than) most of the other values in a set of data. for example in the scores 25,29,3,32,85,33,27,28 both 3 and 85 are "outliers".'''
data1 = plt.figure(figsize=(20, 20))
ax = sns.boxplot(data1=data)
print(ax)
'''remove outliers'''
z = np.abs(stats.zscore(data))
print(z)
'''to check the standard deviation of dataset'''
threshold = 3
print(np.where(z > 3))
'''Quantile divide dataset into 4 groups'''
first_Q1 = data.quantile(0.25)
second_Q2 = data.quantile(0.75)
total = second_Q2 - first_Q1
print(total)
'''imp'''
data = data[(z < 3).all(axis=1)]
print(data)
'''after running above code no. of column and row decraese'''
s = data.shape
print(s)
'''to set now lower bound and upper bound'''
data = data[~((data < (first_Q1 - 1.5 * total)) |(data > (second_Q2 + 1.5 * total))).any(axis=1)]
print(data)
'''after setting lower and upper bound'''
s = data.shape
print(s)
'''after all these check out our outliers is removed or not'''
data2 = plt.figure(figsize=(20, 20))
ax = sns.boxplot(data2=data)
print(ax)
'''feature selection co-relation'''
coolwarm = plt.figure(figsize=(20, 20))
d = sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
print(d)
k = data.describe()
print(k)
'''feature scaling'''
'''Feature Scaling is a technique to standardize the independent features present in the data in a fixed range.'''
StandardScaler = StandardScaler()
'''create dummy variable-dummy variable enble user to used single regression to represent multiple group '''
main_data = pd.get_dummies(data,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
scale_column = ['age','trestbps','chol','thalach','oldpeak']
main_data[scale_column] = StandardScaler.fit_transform(main_data[scale_column])
# read = main_data.describe()
# print(read)
'''now time for model selection we use for k-nearst neighbour model beacause data is more
cluster and combat in this model that useful for more accurate result
'''
Y = main_data['condition']
'''P gives condition variable  and in q drop in condition varibale values'''
X = main_data.drop(['condition'], axis=1)

nd = train_test_split(X,Y,test_size=0.2,random_state=5)
'''before hyperparameter tunning'''
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X, Y)
score = cross_val_score(knn_classifier, X, Y, cv=10)
y_pred_knn = knn_classifier.predict(X)
re = accuracy_score(Y, y_pred_knn)
print(re)
h = score.mean()
print(h)
'''after adding hyperparameter tunning'''
print("This is byk-nearst classifier")
knn_classifier  = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
metric_params=None, n_jobs=1, n_neighbors=5, p=1,
weights='uniform')
knn_classifier.fit(X, Y)
score = cross_val_score(knn_classifier, X, Y, cv=10)
y_pred_knn = knn_classifier.predict(X)
accuracy_score(Y, y_pred_knn)
re1 = accuracy_score(Y, y_pred_knn)
print(re1)
h2 = score.mean()
print(h2)
'''confusion matrix'''
cm = confusion_matrix(Y, y_pred_knn)
plt.title('Heatmap of Confusion Matrix', fontsize=15)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(Y, y_pred_knn))

# print("This is by random forest classifier")
# from sklearn.ensemble import RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
# rf_classifier.fit(X, Y)
# y_pred_rf = rf_classifier.predict(X)
# h3 = accuracy_score(Y, y_pred_rf)
# print(h3)
# score=cross_val_score(rf_classifier,X,Y,cv=10)
# h5 = score.mean()
# print(h5)
#
# '''XGboot classifier'''
#
#
# # xgb_classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
# #        colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
# #        learning_rate=0.1, max_delta_step=0, max_depth=15,
# #        min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
# #        nthread=None, objective='binary:logistic', random_state=23,
# #        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
# #        silent=None, subsample=1, verbosity=1)
# # xgb_classifier.fit(X, Y)
# # y_pred_xgb = xgb_classifier.predict(X)
# # u = accuracy_score(Y, y_pred_xgb)
# # print(u)
#
# ''' AdaBoostClassifier'''
# print('this is by AdaBoostClassifier')
# ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100), n_estimators=100)
# ada_clf.fit(X, Y)
# y_pred_adb = ada_clf.predict(X)
# uh1 = accuracy_score(Y, y_pred_adb)
# print(uh1)
# score=cross_val_score(ada_clf,X,Y,cv=10)
# uh = score.mean()
# print(uh)
#
# print('this is by  GradientBoosting')
# gbc_clf = GradientBoostingClassifier()
# gbc_clf.fit(X, Y)
# y_pred_adb = gbc_clf.predict(X)
# ac = accuracy_score(Y, y_pred_adb)
# print(ac)
# score = cross_val_score(gbc_clf, X, Y, cv=10)
# sc = score.mean()
# print(sc)

'''now save a model'''
## Pickle
from xgboost import XGBClassifier
import pickle

# save model
pickle.dump(knn_classifier, open('Heartmodel.pkl', 'wb'))

# load model
Heart_disease_detector_model = pickle.load(open('Heartmodel.pkl', 'rb'))

# predict the output
y_pred = Heart_disease_detector_model.predict(X)

# confusion matrix
print('Confusion matrix of K – Nearest Neighbor model: \n',confusion_matrix(Y, y_pred),'\n')

# show the accuracy
print('Accuracy of K – Nearest Neighbor  model = ',accuracy_score(Y, y_pred))
