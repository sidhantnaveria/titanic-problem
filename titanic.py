# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:23:22 2020

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
from sklearn.preprocessing import OneHotEncoder as hte
from sklearn import metrics
import pickle
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import  RandomForestClassifier as rfc
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

data= pd.read_csv("C:/Users/sidha/Downloads/titanic/train.csv")
data_new=data.drop(['Ticket','Cabin','PassengerId'],axis=1)

#print(data_new['Age'].isnull().values.sum())
#print(data_new['Age'].median())
data_new['Age'].fillna(int(data_new['Age'].median()),inplace=True)

#data_new.loc[(data_new['Age']>=15) & (data_new['Age']<=30),'ageBtwn15-30']=1
#data_new['ageBtwn15-30'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']>30) & (data_new['Age']<=50),'ageBtwn31-50']=1
#data_new['ageBtwn31-50'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']<15),'ageless15']=1
#data_new['ageless15'].fillna(0,inplace=True)
#
#data_new.loc[(data_new['Age']>50),'agemore50']=1
#data_new['agemore50'].fillna(0,inplace=True)
#data_new=data_new=data_new.drop(['Age'], axis=1)

data_new.loc[data_new['Age'] <= 16, 'Age'] = 0
data_new.loc[(data_new['Age'] > 16) & (data_new['Age']<=32), 'Age'] = 1
data_new.loc[(data_new['Age'] > 32) & (data_new['Age'] <= 48), 'Age'] = 2
data_new.loc[(data_new['Age'] > 48) & (data_new['Age'] <= 64), 'Age'] = 3
data_new.loc[ data_new['Age'] > 64, 'Age'] = 4

#print(data_new.loc[(data_new['Age']>=50)  &(data_new['Sex']=='female')& (data_new['Survived']==1)].count())
#print(data_new[['Age','Survived']].groupby('Age').mean().sort_values(by = 'Survived', ascending = False))


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

#print(data_new['Embarked'].isnull().values.sum())
data_new['Embarked'].fillna('S',inplace=True)
data_new['Embarked']=data_new['Embarked'].astype('category')
data_new['Embarked']=data_new['Embarked'].cat.codes
#data_new= data_new.reset_index(drop=True)
#encoder=ce.OneHotEncoder(cols=['Embarked'])
#data_new=encoder.fit_transform(data_new)
#data_new=data_new.drop(['Embarked_-1'],axis=1)
#
#data_new = pd.get_dummies(data_new, columns=["Pclass"],prefix=['class'] )
#


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


data_new.loc[ data_new['Fare'] <= 7.91, 'Fare'] = 0
data_new.loc[(data_new['Fare'] > 7.91) & (data_new['Fare'] <= 14.454), 'Fare'] = 1
data_new.loc[(data_new['Fare'] > 14.454) & (data_new['Fare'] <= 31), 'Fare']   = 2
data_new.loc[ data_new['Fare'] > 31, 'Fare'] = 3



#data_new.loc[(data_new['SibSp']>=1)  (data_new['Parch']>0),'family' ]=1

data_new['family']=data_new['SibSp']+data_new['Parch']

#print(data_new[['family','Survived']].groupby('family').mean().sort_values(by = 'Survived', ascending = False))

data_new.loc[(data_new['family']==0) ,'familytype' ]=0
data_new.loc[(data_new['family']>0) &(data_new['family']<=3),'familytype' ]=1
data_new.loc[(data_new['family']>3),'familytype' ]=2


#data_new.loc[(data_new['SibSp']==0) & (data_new['Parch']==0),'family' ]=0
data_new=data_new.drop(['SibSp','Parch','family','Name','Sex','Pclass'],axis=1)

#data_new[['Fare']] = preprocessing.StandardScaler().fit_transform(data_new[['Fare']])

X=data_new.drop(['Survived'],axis=1)
Y=data_new['Survived']

#print(data_new[(data_new['Femaleclass1']==1)].count())

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.20,random_state=2)

###########################################################################################################

#model = SGDClassifier()
## fit (train) the classifier
#model.fit(Xtrain, Ytrain)
#
#
#y_test_pridict=model.predict(Xtest)
##with open('C:/Users/sidha/Downloads/titanic/model1.pkl', 'wb') as f:
##    pickle.dump(model, f)
###with open('C:/Users/sidha/Downloads/titanic/model.pkl', 'rb') as f:
###    model2=pickle.load(f)
##    
#y_test_pridict=model.predict(Xtest)
#print(confusion_matrix(Ytest,y_test_pridict))
#print(accuracy_score(Ytest,y_test_pridict))
#print(classification_report(Ytest,y_test_pridict))
#for i in range(1,9):
for i in range(1,20):
    tree=dt(max_depth=i,max_features=0.9)
    tree.fit(Xtrain, Ytrain)
    y_tree_pridict=tree.predict(Xtest)
    print(i)
#    print(confusion_matrix(Ytest,y_tree_pridict))
    print(accuracy_score(Ytest,y_tree_pridict))
#    print(classification_report(Ytest,y_tree_pridict))
#with open('C:/Users/sidha/Downloads/titanic/tree85.4_best.pkl', 'wb') as f:
#    pickle.dump(tree, f)






#rfcmodel=rfc(max_depth=4,min_samples_split=20,n_estimators=70,min_samples_leaf=8,max_features=0.2)
#rfcmodel=rfc(n_estimators= 110, min_samples_split=10, min_samples_leaf= 1, max_features= 0.4, max_depth= 6)
rfcmodel = rfc(max_depth=5,criterion='entropy',n_estimators=18)
rfcmodel.fit(Xtrain, Ytrain)
print(rfcmodel)

y_tree_pridict=rfcmodel.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print(accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))
#with open('C:/Users/sidha/Downloads/titanic/randomtree_latest.pkl', 'wb') as f:
#    pickle.dump(rfcmodel, f)

#xgbclassf=xgb.XGBClassifier(objective="binary:logistic", random_state=42)

#####xgboost########
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic')

params = {
        
        'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],
        'learning_rate': [0.001, 0.01,0.02,0.03, 0.1, 0.2, 0.3,0.4,0.5],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.2,0.3,0.4,0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0,7.0,8.0,9.0, 10.0,11.0,12.0,13.0, 50.0, 100.0],
        'n_estimators': [100]
        }


search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, return_train_score=True)

search.fit(Xtrain, Ytrain)

print(search.best_score_)
print(search.best_params_)

#search.fit(Xtrain, Ytrain)
y_tree_pridict=search.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print("accuracy: ",accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))

#with open('C:/Users/sidha/Downloads/titanic/XGb_80_new.pkl', 'wb') as f:
#    pickle.dump(search.best_estimator_, f)
    
#with open('C:/Users/sidha/Downloads/titanic/XGb_80_new.pkl', 'rb') as f:
#    model2=pickle.load(f)
#    
#y_tree_pridict=model2.predict(Xtest)
#print(confusion_matrix(Ytest,y_tree_pridict))
#print("accuracy: ",accuracy_score(Ytest,y_tree_pridict))
#print(classification_report(Ytest,y_tree_pridict))

##############################################



#rfcmodel=rfc(max_depth=4,min_samples_split=20,n_estimators=100,min_samples_leaf=8,max_features=0.2)
rfcmodel=rfc()

params={
        'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],
        'min_samples_split':[10,15,20,25,30,35,40,45,50,55,60,65],
        'n_estimators':[16,17,18,19,20,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110],
        'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11,12],
        'max_features':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        'criterion':['entropy']
        
        
        }

search2 = RandomizedSearchCV(rfcmodel, param_distributions=params, random_state=42, n_iter=63, cv=4, verbose=1, n_jobs=1, return_train_score=True)

search2.fit(Xtrain, Ytrain)

print(search2.best_score_)
print(search2.best_params_)

y_tree_pridict=search2.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print(accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))
with open('C:/Users/sidha/Downloads/titanic/randamfor_new.pkl', 'wb') as f:
    pickle.dump(search2.best_estimator_, f)

print(Xtrain.columns)
print(data_new.columns)

clf = GaussianNB()

print(clf)
clf.fit(Xtrain, Ytrain)

y_tree_pridict=clf.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print(accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))




eclf = VotingClassifier(estimators=[('lr', search), ('rf', search2)],voting='hard')

#scores = cross_val_score(eclf, Xtrain, Ytrain, scoring='accuracy', cv=5)

eclf.fit(Xtrain, Ytrain)
y_tree_pridict=eclf.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print(accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))

with open('C:/Users/sidha/Downloads/titanic/ensemble2.pkl', 'wb') as f:
    pickle.dump(eclf, f)

with open('C:/Users/sidha/Downloads/titanic/ensemble.pkl', 'rb') as f:
    model2=pickle.load(f)
y_tree_pridict=model2.predict(Xtest)
print(confusion_matrix(Ytest,y_tree_pridict))
print("accuracy: ",accuracy_score(Ytest,y_tree_pridict))
print(classification_report(Ytest,y_tree_pridict))
