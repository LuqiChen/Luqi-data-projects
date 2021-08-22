#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


kf = StratifiedKFold(n_splits=10, random_state=42)


# In[4]:


data_withTFIDF = pd.read_csv("data_withTFIDF_like.csv")
data_withTFIDF = data_withTFIDF.iloc[:, 1:] #delete first column
cdata_lct=data_withTFIDF.drop(['created_at','is_popular' ,'favorite_count','text','retweet_count','clean_text2','words'], axis=1)


# In[5]:


cdata_lct.head()


# In[8]:


data = pd.read_csv("prepared_data_like.csv")
data = data.iloc[:, 1:] #delete first column
cdata_lc  = data.drop(['created_at', 'is_popular','favorite_count','text','retweet_count','clean_text2'], axis=1)


# In[9]:


cdata_lc.head()


# # Logistic regression (calculated features + SMOTE + cross validation)

# In[54]:


y = cdata_lc['is_liked'] 
X = cdata_lc.loc[:, cdata_lc.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[55]:


cross_val_f1_score_lst = []
cross_val_accuracy_lst = []
cross_val_recall_lst = []
cross_val_precision_lst = []
base_val_lst=[]

for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
    # keeping validation set apart and oversampling in each iteration using smote 
    train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
    target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_sample(train, target_train)
    print (X_train_res.shape, y_train_res.shape)
    
    # training the model on oversampled 4 folds of training set
    model = LogisticRegression(C=1e90)
    model.fit(X_train_res, y_train_res)
    # testing on 1 fold of validation set
    validation_preds = model.predict(validation)
    cross_val_recall_lst.append(recall_score(target_val, validation_preds))
    cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
    cross_val_precision_lst.append(precision_score(target_val, validation_preds))
    cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
    base_val_lst.append(target_val.mean())
    
print ('Cross validated accuracy: {:.2f}'.format(np.mean(cross_val_accuracy_lst)))
print ('Cross validated recall score: {:.2f}'.format(np.mean(cross_val_recall_lst)))
print ('Cross validated precision score: {:.2f}'.format(np.mean(cross_val_precision_lst)))
print ('Cross validated f1_score: {:.2f}'.format(np.mean(cross_val_f1_score_lst)))
print ('baseline accuracy: {:.2f}'.format(np.mean(base_val_lst)))


# In[14]:


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
logreg2 = LogisticRegression(C=1e90).fit(X_train_res, y_train_res)


# In[15]:


coef = pd.Series(logreg2.coef_[0], index = X_train.columns)
coef


# In[19]:


imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(2)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model for 'like_count>200' classification (with only calculated features)")


# In[21]:


import statsmodels.api as sm

X_train_new = sm.add_constant(X_train)
logisticmodel = sm.Logit(y_train, X_train_new).fit()

logisticmodel.summary() # get a complete summary of the model


# # Logistic regression (all features + RFE + SMOTE + cross validation)

# In[22]:


y = cdata_lct['is_liked'] 
X = cdata_lct.loc[:, cdata_lct.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[23]:


logreg = LogisticRegression(C=1e90).fit(X_train, y_train)
select = RFE(logreg,1000, step=100) 
selector = select.fit(X_train, y_train)  


# In[24]:


Xnew = selector.transform(X_train) #reduces X to subset identified above
Xnew.shape


# In[25]:


cross_val_f1_score_lst = []
cross_val_accuracy_lst = []
cross_val_recall_lst = []
cross_val_precision_lst = []

for train_index_ls, validation_index_ls in kf.split(Xnew, y_train):
    # keeping validation set apart and oversampling in each iteration using smote 
    train, validation = Xnew[train_index_ls], Xnew[validation_index_ls]
    target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_sample(train, target_train)
    print (X_train_res.shape, y_train_res.shape)
    
    # training the model on oversampled 4 folds of training set
    model = LogisticRegression(C=1e90)
    model.fit(X_train_res, y_train_res)
    # testing on 1 fold of validation set
    validation_preds = model.predict(validation)
    cross_val_recall_lst.append(recall_score(target_val, validation_preds))
    cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
    cross_val_precision_lst.append(precision_score(target_val, validation_preds))
    cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
print ('Cross validated accuracy: {:.2f}'.format(np.mean(cross_val_accuracy_lst)))
print ('Cross validated recall score: {:.2f}'.format(np.mean(cross_val_recall_lst)))
print ('Cross validated precision score: {:.2f}'.format(np.mean(cross_val_precision_lst)))
print ('Cross validated f1_score: {:.2f}'.format(np.mean(cross_val_f1_score_lst)))


# In[26]:


logregnew = LogisticRegression(C=1e90).fit(Xnew, y_train)


# In[29]:


coef = pd.Series(logregnew.coef_[0], index = X_train.columns[selector.get_support()])
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model for 'like_count>200' classification (with all the features)")


# # KNN (calculated features + SMOTE + cross validation)

# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[56]:


y = cdata_lc['is_liked'] 
X = cdata_lc.loc[:, cdata_lc.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[57]:


data=[]
for n in np.arange(1, 25, 2):
    
    model = KNeighborsClassifier(n_neighbors=n)
    cross_val_f1_score_lst = []
    cross_val_accuracy_lst = []
    cross_val_recall_lst = []
    cross_val_precision_lst = []

    for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
        train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
        target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
        # training the model on oversampled 9 folds of training set
        model.fit(X_train_res, y_train_res)
        # testing on 1 fold of validation set
        validation_preds = model.predict(validation)
        cross_val_recall_lst.append(recall_score(target_val, validation_preds))
        cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
        cross_val_precision_lst.append(precision_score(target_val, validation_preds))
        cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
        # keeping validation set apart and oversampling in each iteration using smote 
    n_neighbors= str(n)
    Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
    Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
    Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
    Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

    data.append((n_neighbors, Cross_validated_accuracy,Cross_validated_recall_score,
             Cross_validated_precision_score,Cross_validated_f1_score))
cols=['n_neighbors','Cross validated accuracy','Cross validated recall score',
      'Cross validated precision score','Cross validated f1_score']

result3 = pd.DataFrame(data, columns=cols)


# In[59]:


result3.to_csv("result3.csv",index=False)


# # KNN (all features + SMOTE + cross validation)

# In[34]:


y = cdata_lct['is_liked'] 
X = cdata_lct.loc[:, cdata_lct.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[60]:


data=[]
for n in np.arange(1, 25, 2):
    
    model = KNeighborsClassifier(n_neighbors=n)
    cross_val_f1_score_lst = []
    cross_val_accuracy_lst = []
    cross_val_recall_lst = []
    cross_val_precision_lst = []

    for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
        train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
        target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
        # training the model on oversampled 9 folds of training set
        model.fit(X_train_res, y_train_res)
        # testing on 1 fold of validation set
        validation_preds = model.predict(validation)
        cross_val_recall_lst.append(recall_score(target_val, validation_preds))
        cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
        cross_val_precision_lst.append(precision_score(target_val, validation_preds))
        cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
        # keeping validation set apart and oversampling in each iteration using smote 
    n_neighbors= str(n)
    Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
    Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
    Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
    Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

    data.append((n_neighbors, Cross_validated_accuracy,Cross_validated_recall_score,
             Cross_validated_precision_score,Cross_validated_f1_score))
cols=['n_neighbors','Cross validated accuracy','Cross validated recall score',
      'Cross validated precision score','Cross validated f1_score']

result4 = pd.DataFrame(data, columns=cols)


# In[61]:


result4.to_csv("result4.csv",index=False)


# # Random Forest (calculated features + SMOTE + cross validation)

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


y = cdata_lc['is_liked'] 
X = cdata_lc.loc[:, cdata_lc.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[39]:


data=[]
for n in [5,10,15,20,50,100,150,200,500]:
    
    model = RandomForestClassifier(random_state = 42,n_estimators=n)
    cross_val_f1_score_lst = []
    cross_val_accuracy_lst = []
    cross_val_recall_lst = []
    cross_val_precision_lst = []

    for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
        train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
        target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
        # training the model on oversampled 4 folds of training set
        model.fit(X_train_res, y_train_res)
        # testing on 1 fold of validation set
        validation_preds = model.predict(validation)
        cross_val_recall_lst.append(recall_score(target_val, validation_preds))
        cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
        cross_val_precision_lst.append(precision_score(target_val, validation_preds))
        cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
        # keeping validation set apart and oversampling in each iteration using smote 
    n_estimators= str(n)
    Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
    Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
    Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
    Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

    data.append((n_estimators, Cross_validated_accuracy,Cross_validated_recall_score,
             Cross_validated_precision_score,Cross_validated_f1_score))
cols=['n_estimators','Cross validated accuracy','Cross validated recall score',
      'Cross validated precision score','Cross validated f1_score']

result = pd.DataFrame(data, columns=cols)


# In[40]:


result


# In[41]:


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
RF150 = RandomForestClassifier(random_state = 42,n_estimators=150).fit(X_train_res, y_train_res)
coef = pd.Series(RF150.feature_importances_, index = X_train.columns)
coef


# In[42]:


imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(2)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature Importance in the Random Forest Model for 'like_count>200' classification (with only calculated features)")


# # Random Forest (all features + RFE + SMOTE + cross validation)

# In[43]:


y = cdata_lct['is_liked'] 
X = cdata_lct.loc[:, cdata_lct.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[44]:


data=[]
for n in [5,10,15,20,50,100,150,200,500]:
    
    rf = RandomForestClassifier(random_state = 42,n_estimators=n)
    select = RFE(rf,1000, step=500) # step tells RFE how many features to remove each time model features are evaluated
    selector = select.fit(X_train, y_train) # fit RFE estimator.
    Xnew = selector.transform(X_train) #reduces X to subset identified above
    cross_val_f1_score_lst = []
    cross_val_accuracy_lst = []
    cross_val_recall_lst = []
    cross_val_precision_lst = []

    for train_index_ls, validation_index_ls in kf.split(Xnew, y_train):
        train, validation = Xnew[train_index_ls], Xnew[validation_index_ls]
        target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
        # training the model on oversampled 4 folds of training set
        rf.fit(X_train_res, y_train_res)
        # testing on 1 fold of validation set
        validation_preds = rf.predict(validation)
        cross_val_recall_lst.append(recall_score(target_val, validation_preds))
        cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
        cross_val_precision_lst.append(precision_score(target_val, validation_preds))
        cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
        # keeping validation set apart and oversampling in each iteration using smote 
    n_estimators= str(n)
    Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
    Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
    Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
    Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

    data.append((n_estimators, Cross_validated_accuracy,Cross_validated_recall_score,
             Cross_validated_precision_score,Cross_validated_f1_score))
cols=['n_estimators','Cross validated accuracy','Cross validated recall score',
      'Cross validated precision score','Cross validated f1_score']

result = pd.DataFrame(data, columns=cols)


# In[45]:


result


# In[46]:


rf150 = RandomForestClassifier(random_state = 42,n_estimators=150).fit(X_train, y_train)
select = RFE(rf150,1000, step=100) 
selector = select.fit(X_train, y_train) 
Xnew = selector.transform(X_train)

RF150 = RandomForestClassifier(random_state = 42,n_estimators=150).fit(Xnew, y_train)
coef = pd.Series(RF150.feature_importances_, index = X_train.columns[selector.get_support()])
coef


# In[47]:


imp_coef = pd.concat([coef.sort_values().tail(20),
                     coef.sort_values().head(0)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature Importance in the Random Forest Model for 'like_count>200' classification (with all the features)")


# # SVM (calculated features + SMOTE + cross validation)

# In[48]:


from sklearn.svm import SVC


# In[49]:


y = cdata_lc['is_liked'] 
X = cdata_lc.loc[:, cdata_lc.columns != 'is_liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[50]:


SVCclassifier = SVC(kernel='linear', C=1E10,max_iter=20000)
SVCclassifier.fit(X_train, y_train)
SVCscore = np.mean(cross_val_score(SVCclassifier, X_train, y_train, cv=kf))
print("Mean cross-validation: {:.2f}".format(SVCscore))


# In[51]:


cross_val_f1_score_lst = []
cross_val_accuracy_lst = []
cross_val_recall_lst = []
cross_val_precision_lst = []

for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
    # keeping validation set apart and oversampling in each iteration using smote 
    train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
    target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_sample(train, target_train)
    print (X_train_res.shape, y_train_res.shape)
    
    # training the model on oversampled 4 folds of training set
    model = SVC(kernel='linear', C=1E10,max_iter=20000)
    model.fit(X_train_res, y_train_res)
    # testing on 1 fold of validation set
    validation_preds = model.predict(validation)
    cross_val_recall_lst.append(recall_score(target_val, validation_preds))
    cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
    cross_val_precision_lst.append(precision_score(target_val, validation_preds))
    cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
print ('Cross validated accuracy: {:.2f}'.format(np.mean(cross_val_accuracy_lst)))
print ('Cross validated recall score: {:.2f}'.format(np.mean(cross_val_recall_lst)))
print ('Cross validated precision score: {:.2f}'.format(np.mean(cross_val_precision_lst)))
print ('Cross validated f1_score: {:.2f}'.format(np.mean(cross_val_f1_score_lst)))


# In[53]:


print (y_test.value_counts(normalize= True))


# In[ ]:




