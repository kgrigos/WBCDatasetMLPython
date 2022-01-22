#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scikitplot as skplt
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xlsxwriter


# In[2]:


data = load_breast_cancer()
print (data.feature_names)
print (data.target_names)


# In[3]:


df = pd.read_csv('C:\\Users\\kosta\\Desktop\\Dataset\\data.csv')
df.head(5)


# In[4]:


df.drop(df.columns[[-1, 0]], axis=1, inplace=True)
df.info()


# In[5]:


print ("Total number of diagnosis are ", str(df.shape[0]), ", ", df.diagnosis.value_counts()['B'], "Benign and Malignant are",
       df.diagnosis.value_counts()['M'])


# In[6]:


input= np.loadtxt("C:\\Users\\kosta\\Desktop\\MATI\\Test1.txt",unpack=True)
X=np.array(input)
X=list(X)
print(X)
X=df.iloc[:,X]
print(X)
y = df.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

Gauss = GaussianNB()
Gaussclf = Gauss.fit(X_train, y_train)
Gauss_predicted = Gaussclf.predict(X_test)

Kmeans=KNeighborsClassifier()
Kmeansclf=Kmeans.fit(X_train, y_train)
Kmeans_predicted = Kmeansclf.predict(X_test)

DesTree=DecisionTreeClassifier()
DesTreeclf=DesTree.fit(X_train, y_train)
DesTree_predicted = DesTreeclf.predict(X_test)

Logistic=LogisticRegression()
Logisticclf=Logistic.fit(X_train, y_train)
Logistic_predicted = Logisticclf.predict(X_test)



#print('Accuracy of Decision tree classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
#print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))




# In[7]:


from sklearn.metrics import confusion_matrix


# In[8]:


from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
DesTreetable=classification_report(y_test,DesTree_predicted,output_dict=True)
DesTreeaccuracy=DesTreetable['accuracy']
DesTreefscore=DesTreetable['macro avg']['f1-score']

Kmeanstable=classification_report(y_test,Kmeans_predicted,output_dict=True)
Kmeansaccuracy=Kmeanstable['accuracy']
Kmeansfscore=Kmeanstable['macro avg']['f1-score']


Gausstable=classification_report(y_test,Gauss_predicted,output_dict=True)
Gaussaccuracy=Gausstable['accuracy']
Gaussfscore=Gausstable['macro avg']['f1-score']


Logistictable=classification_report(y_test,Logistic_predicted,output_dict=True)
Logisticaccuracy=Logistictable['accuracy']
Logisticfscore=Logistictable['macro avg']['f1-score']




# In[9]:


DesTree_probs=DesTreeclf.predict_proba(X_test)
DesTree_probs=DesTree_probs[:,1]


Kmeans_probs=Kmeansclf.predict_proba(X_test)
Kmeans_probs=Kmeans_probs[:,1]


Gauss_probs=Gaussclf.predict_proba(X_test)
Gauss_probs=Gauss_probs[:,1]


Logistic_probs=Logisticclf.predict_proba(X_test)
Logistic_probs=Logistic_probs[:,1]


# In[10]:


from sklearn.metrics import roc_curve,roc_auc_score
DesTreeclf_auc=roc_auc_score(y_test,DesTree_probs)
Kmeansclf_auc=roc_auc_score(y_test,Kmeans_probs)
Gaussclf_auc=roc_auc_score(y_test,Gauss_probs)
Logisticclf_auc=roc_auc_score(y_test,Logistic_probs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


Report_Array=[('Guassian Classifier',Gaussfscore,Gaussclf_auc,Gaussaccuracy),
              ('Decision Tree Classifier',DesTreefscore,DesTreeclf_auc,DesTreeaccuracy),
              ('Kmeans Classifier',Kmeansfscore,Kmeansclf_auc,Kmeansaccuracy),
              ('Logistic Regression Classifier',Logisticfscore,Logisticclf_auc,Logisticaccuracy)]


# In[12]:


Report_List=list(Report_Array)


# In[13]:


print(Report_List)


# In[14]:



names=["Guassian Classifier",'Decision Tree Classifier','Kmeans Classifier','Logistic Regression Classifier']
fscore=[Gaussfscore,DesTreefscore,Kmeansfscore,Logisticfscore]
AUC=[Gaussclf_auc,DesTreeclf_auc,Kmeansclf_auc,Logisticclf_auc]
Accuracy=[Gaussaccuracy,DesTreeaccuracy,Kmeansaccuracy,Logisticaccuracy]
# for n, f,auc,acc in zip(names, fscore,AUC,Accuracy):
#     export=print("{} {} {} {}".format(n, f,auc,acc))
# print(export)


# In[15]:


import openpyxl
Export={'Name':["Guassian Classifier",'Decision Tree Classifier','Kmeans Classifier','Logistic Regression Classifier'],
        'F1score':[Gaussfscore,DesTreefscore,Kmeansfscore,Logisticfscore],
        'AUC score':[Gaussclf_auc,DesTreeclf_auc,Kmeansclf_auc,Logisticclf_auc],
        'Test_Set Accuracy':[Gaussaccuracy,DesTreeaccuracy,Kmeansaccuracy,Logisticaccuracy]}
DataExport = pd.DataFrame(Export, columns = ['Name', 'F1score','AUC score','Test_Set Accuracy'])
DataExport.to_excel ('C:\\Users\\kosta\\Desktop\\Dataset\\Test1.xlsx', index = False, header=True)


# In[ ]:




