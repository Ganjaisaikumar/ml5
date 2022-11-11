#!/usr/bin/env python
# coding: utf-8

# In[6]:


#1. Principal Component Analysis
#a. Apply PCA on CC dataset.
#b. Apply k-means algorithm on the PCA result and report your observation if the silhouette scorehas improved or not?
#c. Perform Scaling+PCA+K-Means and report performance.


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC, LinearSVC
import seaborn as sns
from sklearn.cluster import KMeans
sns.set(style="white", color_codes=True)
from sklearn import metrics
import warnings
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
df= pd.read_csv("C:\\Users\\sai\\Downloads\\datasets(1)\\datasets\\CC.csv")
df.head()
print(df.shape)

df['TENURE'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.
A = df.iloc[:,[1,2,3,4]]#integer-location based indexing for selection by position.
B = df.iloc[:,-1]

label_encoder = preprocessing.LabelEncoder()
df['CUST_ID'] = label_encoder.fit_transform(df.CUST_ID.values)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(A)

principalDataframe = pd.DataFrame(data = principalComponents, columns = [' p_component 1', 'p_component 2'])
#Two-dimensional, size-mutable, potentially heterogeneous tabular data.

finalDataframe = pd.concat([principalDataframe, df[['TENURE']]], axis = 1)
print(finalDataframe.head())

nclusters = 2 
kmean = KMeans(n_clusters=nclusters)
kmean.fit(A)

y_cluster_kmeans = kmean.predict(A)

score = metrics.silhouette_score(A, y_cluster_kmeans)
print(score)
scaler = StandardScaler()
X_Scale = scaler.fit_transform(A)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_Scale)

principalDf1 = pd.DataFrame(data = principalComponents, columns = [' p_component 1', ' p_component 2'])

finalDf1 = pd.concat([principalDf1, df[['TENURE']]], axis = 1)
print(finalDf1.head())

nclusters = 2 
km = KMeans(n_clusters=nclusters)
print(km.fit(X_Scale))

y_cluster_kmeans = km.predict(X_Scale)

silhouette_score = metrics.silhouette_score(X_Scale, y_cluster_kmeans)
print(silhouette_score)


# In[8]:


#2. Use pd_speech_features.csv
#a. Perform Scaling
#b. Apply PCA (k=3)
#c. Use SVM to report performance
from sklearn.model_selection import train_test_split
df= pd.read_csv("C:\\Users\\sai\\Downloads\\datasets(1)\\datasets\\pd_speech_features.csv")
df.head()
print(df.shape)
print(df['class'].value_counts())
A = df.drop('class',axis=1).values
B = df['class'].values
scaler = StandardScaler()
A_Scale = scaler.fit_transform(A)
pca2 = PCA(n_components=3)
principal_Component = pca2.fit_transform(A_Scale)

pDf = pd.DataFrame(data = principal_Component, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDataframe = pd.concat([pDf, df[['class']]], axis = 1)
print(finalDataframe.head())
X_train, X_test, y_train, y_test = train_test_split(A_Scale,B, test_size=0.3,random_state=0)
svc = SVC(max_iter=1000)
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)

accuracy_svc = round(svc.score(X_train, y_train) * 100, 2)

print("svm accuracy =", accuracy_svc)


# In[9]:


#3. Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data to k=2. 
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
df = pd.read_csv("C:\\Users\\sai\\Downloads\\datasets(1)\\datasets\\Iris.csv")
print(df.head())
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(df.iloc[:,range(0,4)].values)
class_le = LabelEncoder()
y = class_le.fit_transform(df['Species'].values)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y)
data=pd.DataFrame(X_train_lda)
data['class']=y
data.columns=["LD1","LD2","class"]
print(data.head())
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
print(plt.show())


# In[10]:


#4. Briefly identify the difference between PCA and LDA

    #PCA:principal component analysis.
    #LDA: linear discriminant analysis.
    #PCA and LDA are two popular dimensionality reduction methods commonly used on data with too many input features
    #Principal Component Analysis (PCA) for short is a commonly used unsupervised linear transformation technique. PCA reduces the number of dimensions by finding the maximum variance in high dimensional data.
    #Linear Discriminant Analysis or LDA for short is a supervised method that takes class labesl into account when reducing the number of dimensions. The goal of LDA is to find a feature subspace that best optimizes class separability.


# In[ ]:




