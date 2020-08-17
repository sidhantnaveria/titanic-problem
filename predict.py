# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:32:43 2020

@author: sidhant
"""

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mse
from sklearn.metrics import r2_score,classification_report, confusion_matrix,accuracy_score
from sklearn import metrics
import pickle
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import  RandomForestClassifier as rfc
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV


data= pd.read_csv("C:/Users/sidha/Downloads/titanic/test.csv")

data_new=data.drop(['Ticket','Cabin','PassengerId'],axis=1)

print(data_new['Fare'].isnull().values.sum())
data_new['Age'].fillna(int(data_new['Age'].median()),inplace=True)
#
#data_new.loc[(data_new['Age']>=15) & (data_new['Age']<=30),'ageBtwn15-30']=1
#data_new['ageBtwn15-30'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']>30) & (data_new['Age']<=50),'ageBtwn31-50']=1
#data_new['ageBtwn31-50'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']<15),'ageless15']=1
#data_new['ageless15'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']<50),'agemore50']=1
#data_new['agemore50'].fillna(0,inplace=True)
#data_new=data_new=data_new.drop(['Age'], axis=1)

data_new.loc[data_new['Age'] <= 16, 'Age'] = 0
data_new.loc[(data_new['Age'] > 16) & (data_new['Age']<=32), 'Age'] = 1
data_new.loc[(data_new['Age'] > 32) & (data_new['Age'] <= 48), 'Age'] = 2
data_new.loc[(data_new['Age'] > 48) & (data_new['Age'] <= 64), 'Age'] = 3
data_new.loc[ data_new['Age'] > 64, 'Age'] = 4


data_new['Title'] = data_new['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

data_new['Title'] = data_new['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')
data_new['Title'] = data_new['Title'].replace(['Jonkheer', 'Master'], 'Master')
data_new['Title'] = data_new['Title'].replace(['Don', 'Sir', 'Countess', 'Lady', 'Dona'], 'Royalty')
data_new['Title'] = data_new['Title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
data_new['Title'] = data_new['Title'].replace(['Mlle', 'Miss'], 'Miss')
  

# Imputing missing values with 0
data_new['Title'] = data_new['Title'].fillna(0)
data_new['Title']=data_new['Title'].astype('category')
data_new['Title']=data_new['Title'].cat.codes

#data_new = pd.get_dummies(data_new, columns=["Title"],prefix=['Title'] )


#data_new = pd.get_dummies(data_new, columns=["Pclass"],prefix=['class'] )

data_new['Fare'].fillna(int(data_new['Fare'].median()),inplace=True)

data_new.loc[ data_new['Fare'] <= 7.91, 'Fare'] = 0
data_new.loc[(data_new['Fare'] > 7.91) & (data_new['Fare'] <= 14.454), 'Fare'] = 1
data_new.loc[(data_new['Fare'] > 14.454) & (data_new['Fare'] <= 31), 'Fare']   = 2
data_new.loc[ data_new['Fare'] > 31, 'Fare'] = 3


#encoder=ce.OneHotEncoder(cols=['Embarked'])
#data_new=encoder.fit_transform(data_new)
#data_new=data_new.drop(['Embarked_-1'],axis=1)

data_new['Embarked']=data_new['Embarked'].astype('category')
data_new['Embarked']=data_new['Embarked'].cat.codes

data_new['Sex'].loc[data_new['Sex']=='male']=0
data_new['Sex'].loc[data_new['Sex']=='female']=1

data_new.loc[(data_new['Sex']==1)&(data_new['Pclass']==1),'Femaleclass1']=1
data_new['Femaleclass1'].fillna(0,inplace=True)

data_new.loc[(data_new['Sex']==1)&(data_new['Pclass']==2),'Femaleclass2']=1
data_new['Femaleclass2'].fillna(0,inplace=True)

data_new.loc[(data_new['Sex']==1)&(data_new['Pclass']==3),'Femaleclass3']=1
data_new['Femaleclass3'].fillna(0,inplace=True)

data_new.loc[(data_new['Sex']==0)&(data_new['Pclass']==1),'maleclass1']=1
data_new['maleclass1'].fillna(0,inplace=True)

data_new.loc[(data_new['Sex']==0)&(data_new['Pclass']==2),'maleclass2']=1
data_new['maleclass2'].fillna(0,inplace=True)

data_new.loc[(data_new['Sex']==0)&(data_new['Pclass']==3),'maleclass3']=1
data_new['maleclass3'].fillna(0,inplace=True)

data_new['family']=data_new['SibSp']+data_new['Parch']


data_new.loc[(data_new['family']==0) ,'familytype' ]=0
data_new.loc[(data_new['family']>0) &(data_new['family']<=3),'familytype' ]=1
data_new.loc[(data_new['family']>3),'familytype' ]=2


data_new=data_new.drop(['SibSp','Parch','family','Name','Sex','Pclass'],axis=1)


#data_new[['Fare']] = preprocessing.StandardScaler().fit_transform(data_new[['Fare']])



with open('C:/Users/sidha/Downloads/titanic/randomtree_latest.pkl', 'rb') as f:
    model2=pickle.load(f)
    
y_tree_pridict=model2.predict(data_new)

pred=data.PassengerId
Survival=pd.Series(y_tree_pridict)
final= { 'PassengerId': pred, 'Survived': Survival } 

out=pd.DataFrame(final)
out.to_csv('C:/Users/sidha/Downloads/titanic/prediction1.csv',index=False)


