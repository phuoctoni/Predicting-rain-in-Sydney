#!/usr/bin/env python
# coding: utf-8

# # Predicting rain in Sydney
# 
# Dataset link: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

# In[151]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# # Load dataset as data frame

# In[152]:


weather_df = pd.read_csv('weatherAUS.csv')


# # Filter the data frame to get data for Sydney

# In[153]:


sydney_df = weather_df[weather_df['Location'] == 'Sydney']
sydney_df.info()


# # Let's see how the data looks like

# In[154]:


sydney_df.head(10)


# # Check percentage of yes and no in target column

# In[156]:


rain_counts = sydney_df['RainTomorrow'].value_counts()
plt.pie(rain_counts, autopct = '%1.2f%%', labels=sydney_df.RainTomorrow.unique())
plt.title('Percentage of rain and not rain ')
plt.show()


# # Drop irrelevant attributes

# In[ ]:


sydney_df = sydney_df.drop(columns = ['RISK_MM','Date','Location'], axis=1)


# # Calculate total of null values in each numerical data

# In[ ]:


sydney_df.select_dtypes(exclude=['object']).isnull().sum()


# # Replace all null values with means

# In[ ]:


wind_speed_mean = np.mean(sydney_df['WindGustSpeed'])
sydney_df['WindGustSpeed'] = sydney_df['WindGustSpeed'].fillna(wind_speed_mean)
cloud_9am_mean = np.mean(sydney_df['Cloud9am'])
sydney_df['Cloud9am'] = sydney_df['Cloud9am'].fillna(cloud_9am_mean)
cloud_3pm_mean = np.mean(sydney_df['Cloud3pm'])
sydney_df['Cloud3pm'] = sydney_df['Cloud3pm'].fillna(cloud_3pm_mean)
maxtemp_mean = np.mean(sydney_df['MaxTemp'])
sydney_df['MaxTemp'] = sydney_df['MaxTemp'].fillna(maxtemp_mean)
mintemp_mean = np.mean(sydney_df['MinTemp'])
sydney_df['MinTemp'] = sydney_df['MinTemp'].fillna(mintemp_mean)
temp9am_mean = np.mean(sydney_df['Temp9am'])
sydney_df['Temp9am'] = sydney_df['Temp9am'].fillna(temp9am_mean)
temp3pm_mean = np.mean(sydney_df['Temp3pm'])
sydney_df['Temp3pm'] = sydney_df['Temp3pm'].fillna(temp3pm_mean)
humid9am_mean = np.mean(sydney_df['Humidity9am'])
sydney_df['Humidity9am'] = sydney_df['Humidity9am'].fillna(humid9am_mean)
humid3pm_mean = np.mean(sydney_df['Humidity3pm'])
sydney_df['Humidity3pm'] = sydney_df['Humidity3pm'].fillna(humid3pm_mean)
sun_mean = np.mean(sydney_df['Sunshine'])
sydney_df['Sunshine'] = sydney_df['Sunshine'].fillna(sun_mean)
press9am_mean = np.mean(sydney_df['Pressure9am'])
sydney_df['Pressure9am'] = sydney_df['Pressure9am'].fillna(press9am_mean)
press3pm_mean = np.mean(sydney_df['Pressure3pm'])
sydney_df['Pressure3pm'] = sydney_df['Pressure3pm'].fillna(press3pm_mean)
wind9am_mean = np.mean(sydney_df['WindSpeed9am'])
sydney_df['WindSpeed9am'] = sydney_df['WindSpeed9am'].fillna(wind9am_mean)
wind3pm_mean = np.mean(sydney_df['WindSpeed3pm'])
sydney_df['WindSpeed3pm'] = sydney_df['WindSpeed3pm'].fillna(wind3pm_mean)
eva_mean = np.mean(sydney_df['Evaporation'])
sydney_df['Evaporation'] = sydney_df['Evaporation'].fillna(eva_mean)
rainfall_mean = np.mean(sydney_df['Rainfall'])
sydney_df['Rainfall'] = sydney_df['Rainfall'].fillna(rainfall_mean)


# # Calculate total of null values in each categorical data

# In[ ]:


sydney_df.select_dtypes(include=['object']).isnull().sum()


# # Replace null values with modes

# In[ ]:


sydney_df['WindGustDir'] = sydney_df['WindGustDir'].fillna(sydney_df['WindGustDir'].value_counts().index[0])
sydney_df['WindDir9am'] = sydney_df['WindDir9am'].fillna(sydney_df['WindDir9am'].value_counts().index[0])
sydney_df['WindDir3pm'] = sydney_df['WindDir3pm'].fillna(sydney_df['WindDir3pm'].value_counts().index[0])
sydney_df['RainToday'] = sydney_df['RainToday'].fillna(sydney_df['RainToday'].value_counts().index[0])
sydney_df['RainTomorrow'] = sydney_df['RainTomorrow'].fillna(sydney_df['RainTomorrow'].value_counts().index[0])


# # As you see below, there is no more null value in my dataset

# In[ ]:


sydney_df.isnull().sum()


# In[ ]:


target = sydney_df['RainTomorrow']
sydney_df.drop('RainTomorrow', axis=1, inplace=True)


# # There are 2 types of data in my dataset: numerical data and categorical data

# In[ ]:


sydney_df.dtypes


# # Divide the dataset into 2 groups: numerical variables and categorical variables

# In[ ]:


cat_vars = list(sydney_df.select_dtypes(include='object').columns)
num_vars = list(sydney_df.select_dtypes(exclude='object').columns)


# # Next, combine numerical and categorical attributes into a single transformer and save in a new data frame

# In[ ]:


columns_trans = ColumnTransformer([('num',StandardScaler(),num_vars),('cat',OrdinalEncoder(),cat_vars)])


# In[ ]:


sydney_df_trans = columns_trans.fit_transform(sydney_df)


# In[ ]:


sydney_df_trans.shape


# # Then I use label encoder to encode my target variable, 0=No and 1=Yes.

# In[ ]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)
y


# # I split my data into train set and test set with 75%-25% split.

# In[ ]:


X = sydney_df_trans
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


# # After that, I evaluate accuracy of training sets for Logistic regression, Knn, Decicion tree, Gaussian NB and SVM, respectively with 5 fold cross validation.

# In[ ]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "Accuracy: %s--> %f +/- %f" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# # As you see above, LR, GaussianNB and SVM have over 0.8 accuracy so I pick LR, GaussianNB and SVM to train these models then make predictions on test sets

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# # - Accuracy of Logistic Regression on test set is 0.82
# # - F1-score for predicting Yes and No is 0.57 and 0.89

# In[ ]:


svm = SVC(probability=True)
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# # - Accuracy of SVM on test set is 0.83
# # - F1-score for predicting Yes and No is 0.54 and 0.89

# In[ ]:


gaussianNB = GaussianNB()
gaussianNB.fit(X_train, Y_train)
predictions = gaussianNB.predict(X_test)
print("Accuracy: ",accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# # - Accuracy of Gaussian NB on test set is 0.79
# # - F1-score for predicting Yes and No is 0.58 and 0.86

# # Build ROC curve and calculate AUC for LR, SVM and GaussianNB

# In[ ]:


y_scores_lr = cross_val_predict(lr, X, y, method='predict_proba',cv=5)
y_scores_svm = cross_val_predict(svm, X, y, method='predict_proba',cv=5)
y_scores_nb = cross_val_predict(gaussianNB, X, y, method='predict_proba',cv=5)
fpr_lr, tpr_lr, threshold_lr = roc_curve(y,y_scores_lr[:,1])
fpr_svm, tpr_svm, threshold_svm = roc_curve(y,y_scores_svm[:,1])
fpr_nb, tpr_nb, threshold_nb = roc_curve(y,y_scores_nb[:,1])
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_nb, tpr_nb, label='GaussianNB')
plt.legend()
plt.show()


# In[ ]:


print ('LR: ', roc_auc_score(y,y_scores_lr[:,1]))
print ('SVM: ', roc_auc_score(y,y_scores_svm[:,1]))
print ('GaussianNB: ', roc_auc_score(y,y_scores_nb[:,1]))


# # Overall, after comparing accuracy score, F1-score and AUC among all 3 models, Logistic regression, Support vector machine and Gaussian Naive Bayes, Logistic Regression is the best model to predict whether Sydney will have rain the next day or not.
