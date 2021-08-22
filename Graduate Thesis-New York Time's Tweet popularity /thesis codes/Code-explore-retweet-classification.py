#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.feature_selection import RFE


# In[13]:


kf = StratifiedKFold(n_splits=10, random_state=42)


# In[14]:


data_withTFIDF = pd.read_csv("data_withTFIDF.csv")
data_withTFIDF = data_withTFIDF.iloc[:, 1:] #delete first column


# In[15]:


cdata_rct=data_withTFIDF.drop(['created_at', 'favorite_count','text','retweet_count','clean_text2','words'], axis=1)


# In[16]:


cdata_rct.head()


# In[17]:


cdata_rct.shape


# In[18]:


ldata = pd.read_csv("prepared_data.csv")
ldata = ldata.iloc[:, 1:] #delete first column
ldata.head()


# In[19]:


ldata_rc2  = ldata.drop(['created_at', 'favorite_count','text','retweet_count','clean_text2'], axis=1)
ldata_rc2.head()


# # Logistic regression with all the features & cross validation

# In[7]:


y = cdata_rct["is_popular"]
X = cdata_rct.loc[:, cdata_rct.columns != 'is_popular']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[9]:


logreg = LogisticRegression(C=1e90).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# In[10]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)


# In[11]:


print("Cross validated accuracy: {:.2f}".format(np.mean(cross_val_score(logreg, X_train, y_train, cv=kf))))
print ('Cross validated recall score: {:.2f}'.format(np.mean(cross_val_score(logreg, X_train, y_train, cv=kf,scoring='recall'))))
print ('Cross validated precision score: {:.2f}'.format(np.mean(cross_val_score(logreg, X_train, y_train, cv=kf,scoring='precision'))))
print ('Cross validated f1_score: {:.2f}'.format(np.mean(cross_val_score(logreg, X_train, y_train, cv=kf,scoring='f1'))))


# In[12]:


#Baseline
#print (y_test.mean())
print (y_test.value_counts(normalize=True))


# In[13]:


coef = pd.Series(logreg.coef_[0], index = X_train.columns)


# In[14]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model")


# # Logistic regression with all the features & cross validation # SMOTE

# In[20]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[16]:


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


# In[17]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model")


# # Logistic regression (all features + RFE + SMOTE + cross validation)

# In[19]:


# import seaborn as sns
# corr = cdata_rct.corr()
# ax = sns.heatmap(
#     corr, 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );


# In[20]:


logreg = LogisticRegression(C=1e90).fit(X_train, y_train)
select = RFE(logreg,1000, step=100) # step tells RFE how many features to remove each time model features are evaluated
selector = select.fit(X_train, y_train) # fit RFE estimator.


# In[21]:


print("Num Features: %d" % selector.n_features_)


# In[22]:


Xnew = selector.transform(X_train) #reduces X to subset identified above
Xnew.shape


# In[23]:


type(Xnew)


# In[24]:


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


# In[25]:


logregnew = LogisticRegression(C=1e90).fit(Xnew, y_train)


# In[26]:


coef = pd.Series(logregnew.coef_[0], index = X_train.columns[selector.get_support()])
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model")


# In[27]:


#print ('coefficients',selector.estimator_.coef_)


# # Logistic regression (calculated features + SMOTE + cross validation)

# In[33]:


y = ldata_rc2['is_popular'] 
X = ldata_rc2.loc[:, ldata_rc2.columns != 'is_popular']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[35]:


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


# In[33]:


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
logreg2 = LogisticRegression(C=1e90).fit(X_train_res, y_train_res)


# In[34]:


coef = pd.Series(logreg2.coef_[0], index = X_train.columns)


# In[35]:


coef


# In[37]:


imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(2)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Logistic Model for 'retweet count>100'classification (with only calculated features)")


# In[37]:


import statsmodels.api as sm

X_train_new = sm.add_constant(X_train)
logisticmodel = sm.Logit(y_train, X_train_new).fit()

logisticmodel.summary() # get a complete summary of the model


# # KNN (calculated features + SMOTE + cross validation)

# In[38]:


from sklearn.neighbors import KNeighborsClassifier


# In[39]:


y = ldata_rc2['is_popular'] 
X = ldata_rc2.loc[:, ldata_rc2.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[40]:


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
    
    
        # training the model on oversampled 4 folds of training set
        model.fit(X_train_res, y_train_res)
        # testing on 1 fold of validation set
        validation_preds = model.predict(validation)
        cross_val_recall_lst.append(recall_score(target_val, validation_preds))
        cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
        cross_val_precision_lst.append(precision_score(target_val, validation_preds))
        cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
        # keeping validation set apart and oversampling in each iteration using smote 
    print('when n_neighbors='+ str(n))
    print ('Cross validated accuracy: {:.2f}'.format(np.mean(cross_val_accuracy_lst)))
    print ('Cross validated recall score: {:.2f}'.format(np.mean(cross_val_recall_lst)))
    print ('Cross validated precision score: {:.2f}'.format(np.mean(cross_val_precision_lst)))
    print ('Cross validated f1_score: {:.2f}'.format(np.mean(cross_val_f1_score_lst)))


# In[40]:


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

result1 = pd.DataFrame(data, columns=cols)


# In[42]:


result1.to_csv("result1.csv",index=False)


# In[ ]:


# ax = plt.gca()

# df.plot(kind='line',x='name',y='num_children',ax=ax)
# df.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)

# plt.show()


# In[43]:


# n_neighbors = 15
# knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
# coef = pd.Series(knn.feature_importance, index = X_train.columns)


# # KNN (all features + SMOTE + cross validation)

# In[43]:


y = cdata_rct["is_popular"]
X = cdata_rct.loc[:, cdata_rct.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[45]:


#knn = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)


# In[44]:


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

result2 = pd.DataFrame(data, columns=cols)


# In[46]:


result2.to_csv("result2.csv",index=False)


# # Random Forest (calculated features + SMOTE + cross validation)

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


y = ldata_rc2['is_popular'] 
X = ldata_rc2.loc[:, ldata_rc2.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[50]:


# data=[]
# for n in [5,10,15,20,50,100,150,200,500]:
    
#     model = RandomForestClassifier(random_state = 42,n_estimators=n)
#     cross_val_f1_score_lst = []
#     cross_val_accuracy_lst = []
#     cross_val_recall_lst = []
#     cross_val_precision_lst = []

#     for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
#         train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
#         target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
#         sm = SMOTE(random_state=42)
#         X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
#         # training the model on oversampled 4 folds of training set
#         model.fit(X_train_res, y_train_res)
#         # testing on 1 fold of validation set
#         validation_preds = model.predict(validation)
#         cross_val_recall_lst.append(recall_score(target_val, validation_preds))
#         cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
#         cross_val_precision_lst.append(precision_score(target_val, validation_preds))
#         cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
#         # keeping validation set apart and oversampling in each iteration using smote 
#     n_estimators= str(n)
#     Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
#     Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
#     Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
#     Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

#     data.append((n_estimators, Cross_validated_accuracy,Cross_validated_recall_score,
#              Cross_validated_precision_score,Cross_validated_f1_score))
# cols=['n_estimators','Cross validated accuracy','Cross validated recall score',
#       'Cross validated precision score','Cross validated f1_score']

# result = pd.DataFrame(data, columns=cols)


# In[51]:


#result


# In[52]:


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
RF100 = RandomForestClassifier(random_state = 42,n_estimators=100).fit(X_train_res, y_train_res)
coef = pd.Series(RF100.feature_importances_, index = X_train.columns)
coef


# In[53]:


imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(2)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature Importance in the Random Forest Model (with only calculated features)")


# # Random Forest (all features + RFE + SMOTE + cross validation)

# In[54]:


y = cdata_rct["is_popular"]
X = cdata_rct.loc[:, cdata_rct.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[55]:


# data=[]
# for n in [5,10,15,20,50,100,150,200,500]:
    
#     rf = RandomForestClassifier(random_state = 42,n_estimators=n)
#     select = RFE(rf,1000, step=500) # step tells RFE how many features to remove each time model features are evaluated
#     selector = select.fit(X_train, y_train) # fit RFE estimator.
#     Xnew = selector.transform(X_train) #reduces X to subset identified above
#     cross_val_f1_score_lst = []
#     cross_val_accuracy_lst = []
#     cross_val_recall_lst = []
#     cross_val_precision_lst = []

#     for train_index_ls, validation_index_ls in kf.split(Xnew, y_train):
#         train, validation = Xnew[train_index_ls], Xnew[validation_index_ls]
#         target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
#         sm = SMOTE(random_state=42)
#         X_train_res, y_train_res = sm.fit_sample(train, target_train)
    
    
#         # training the model on oversampled 4 folds of training set
#         rf.fit(X_train_res, y_train_res)
#         # testing on 1 fold of validation set
#         validation_preds = rf.predict(validation)
#         cross_val_recall_lst.append(recall_score(target_val, validation_preds))
#         cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
#         cross_val_precision_lst.append(precision_score(target_val, validation_preds))
#         cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
#         # keeping validation set apart and oversampling in each iteration using smote 
#     n_estimators= str(n)
#     Cross_validated_accuracy = '{:.2f}'.format(np.mean(cross_val_accuracy_lst))
#     Cross_validated_recall_score='{:.2f}'.format(np.mean(cross_val_recall_lst))
#     Cross_validated_precision_score= '{:.2f}'.format(np.mean(cross_val_precision_lst))
#     Cross_validated_f1_score='{:.2f}'.format(np.mean(cross_val_f1_score_lst))

#     data.append((n_estimators, Cross_validated_accuracy,Cross_validated_recall_score,
#              Cross_validated_precision_score,Cross_validated_f1_score))
# cols=['n_estimators','Cross validated accuracy','Cross validated recall score',
#       'Cross validated precision score','Cross validated f1_score']

# result = pd.DataFrame(data, columns=cols)


# In[56]:


#result


# In[57]:


#result.to_csv("result_random forest all features.csv")


# In[58]:


rf50 = RandomForestClassifier(random_state = 42,n_estimators=50).fit(X_train, y_train)
select = RFE(rf50,1000, step=100) 
selector = select.fit(X_train, y_train) 
Xnew = selector.transform(X_train)

RF50 = RandomForestClassifier(random_state = 42,n_estimators=50).fit(Xnew, y_train)
coef = pd.Series(RF50.feature_importances_, index = X_train.columns[selector.get_support()])
coef


# In[59]:


imp_coef = pd.concat([coef.sort_values().tail(20),
                     coef.sort_values().head(0)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature Importance in the Random Forest Model (with all features after dimension reduction)")


# # SVM (calculated features + SMOTE + cross validation)

# In[5]:


from sklearn.svm import SVC


# In[6]:


y = ldata_rc2['is_popular'] 
X = ldata_rc2.loc[:, ldata_rc2.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 


# In[7]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[11]:


SVCclassifier = SVC(kernel='linear', C=1E10,max_iter=20000)
SVCclassifier.fit(X_train, y_train)
SVCscore = np.mean(cross_val_score(SVCclassifier, X_train, y_train, cv=kf))
print("Mean cross-validation: {:.2f}".format(SVCscore))


# In[21]:


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


# # SVM (all features + SMOTE + cross validation+RFE)

# In[32]:


# from sklearn.preprocessing import MinMaxScaler

# y = cdata_rct["is_popular"]
# X = cdata_rct.loc[:, cdata_rct.columns != 'is_popular']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
# scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)

# X_train_scaled=scaling.transform(X_train)
# X_test_scaled=scaling.transform(X_test)

# SVCclassifier = SVC(kernel='linear', C=1E10,max_iter=20000).fit(X_train_scaled, y_train)
# select = RFE(SVCclassifier,1000, step=100) # step tells RFE how many features to remove each time model features are evaluated
# selector = select.fit(X_train_scaled, y_train) # fit RFE estimator.
# Xnew = selector.transform(X_train_scaled) #reduces X to subset identified above
# Xnew.shape


# # Logistic regression (scaled data+calculated features + SMOTE + cross validation)

# In[22]:


y = ldata_rc2['is_popular'] 
X = ldata_rc2.loc[:, ldata_rc2.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[24]:


cross_val_f1_score_lst = []
cross_val_accuracy_lst = []
cross_val_recall_lst = []
cross_val_precision_lst = []

for train_index_ls, validation_index_ls in kf.split(X_train_scaled, y_train):
    # keeping validation set apart and oversampling in each iteration using smote 
    train, validation = X_train_scaled[train_index_ls], X_train_scaled[validation_index_ls]
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


# # Logistic regression all features (scaled data+RFE+ SMOTE + cross validation)

# In[26]:


y = cdata_rct["is_popular"]
X = cdata_rct.loc[:, cdata_rct.columns != 'is_popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[27]:


logreg = LogisticRegression(C=1e90).fit(X_train_scaled, y_train)
select = RFE(logreg,1000, step=100) # step tells RFE how many features to remove each time model features are evaluated
selector = select.fit(X_train_scaled, y_train) # fit RFE estimator.
Xnew = selector.transform(X_train_scaled) #reduces X to subset identified above
Xnew.shape


# In[28]:


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


# In[ ]:




