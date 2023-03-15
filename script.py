#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[3]:


data = pd.read_csv("Bank_loans.csv")
df = data.copy()
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df[df["Loan ID"].isna()]


# In[8]:


# drop useless data from dataset

df = df[~df["Loan ID"].isna()]


# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


df.describe().T


# In[12]:


plt.figure(figsize = (10,8))
sns.boxplot(df["Credit Score"]);


# In[13]:


df[df["Credit Score"] > 5000]


# In[14]:


df["Credit Score"].describe()


# In[15]:


df["Credit Score"] = df["Credit Score"].fillna(df["Credit Score"].median())


# In[16]:


df.info()


# In[ ]:





# In[17]:


df["Annual Income"].describe()


# In[18]:


df["Annual Income"].describe().apply(lambda x:  ("%.1f" % x))


# In[19]:


df[df["Annual Income"] > 100000000]


# In[20]:


# clean outliers
df = df[ ~ (df["Annual Income"] > 100000000)]
df["Annual Income"].describe().apply(lambda x:  ("%.1f" % x))


# In[21]:


plt.figure(figsize = (10,8))
sns.boxplot(df["Annual Income"]);


# In[22]:


df["Annual Income"]


# In[23]:


plt.figure(figsize = (10,8))
sns.distplot(df["Annual Income"]);


# In[24]:


df["Annual Income"] = df["Annual Income"].fillna(df["Annual Income"].median())


# In[25]:


plt.figure(figsize = (10,8))
sns.distplot(df["Annual Income"]);


# In[26]:


df.info()


# In[27]:


df["Years in current job"].unique()


# In[ ]:





# In[28]:


# convert to float values

df["Years in current job"] = df["Years in current job"].str.extract('(\d+)').astype(float)
df["Years in current job"]


# In[29]:


df["Years in current job"].describe()


# In[ ]:





# In[30]:


df["Years in current job"] = df["Years in current job"].fillna(0)
df["Years in current job"]


# In[ ]:





# In[31]:


sns.distplot(df["Years in current job"]);


# In[32]:


df.info()


# In[ ]:





# In[33]:


df["Months since last delinquent"].describe()


# In[34]:


plt.figure(figsize = (10,8))
sns.distplot(df["Months since last delinquent"]);


# In[35]:


# drop variable because there are lots of missing value

df = df.drop("Months since last delinquent", axis = 1)
df.head()


# In[36]:


df.info()


# In[37]:


df.dropna(subset = ["Maximum Open Credit","Bankruptcies","Tax Liens"], inplace = True)


# In[38]:


df.info()


# In[39]:


df.drop(["Loan ID", "Customer ID"], axis = 1, inplace = True)
df.head()


# In[ ]:





# # Resolving imbalance problem

# In[40]:


from sklearn.utils import resample


# In[41]:


df["Loan Status"].value_counts()


# In[42]:


df_majority = df[df["Loan Status"] == "Fully Paid"]
df_minority = df[df["Loan Status"] == "Charged Off"]


# In[43]:


df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=77207,    # to match majority class
                                 random_state=123) # reproducible results


# In[44]:


df_minority_upsampled


# In[45]:


df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# In[46]:


df_upsampled["Loan Status"].value_counts()


# In[ ]:





# # Feature Engineering

# In[47]:


df_upsampled.select_dtypes("object").apply(lambda x: x.unique())


# In[48]:


df_upsampled["Purpose"].unique()


# In[49]:


df_upsampled["Loan Status"] = df_upsampled["Loan Status"].replace({"Fully Paid" : 1, "Charged Off" : 0})


# In[50]:


df_upsampled.head()


# In[51]:


df_upsampled.head()


# In[ ]:





# In[52]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


# In[53]:


X = df_upsampled.drop("Loan Status", axis = 1)
y = df_upsampled["Loan Status"]


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[55]:


X_train.shape


# In[56]:


X_test.shape


# In[57]:


X_train.head()


# In[ ]:





# ## Preprocessing

# In[58]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer


# In[59]:


def map_func(x):
    return x.str.extract('(\d+)').astype(float)


class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        X_scaled = X.copy()
        X_scaled[self.columns] = map_func(X_scaled[self.columns])
        return X_scaled
    
    def fit(self, X, y=None, **fit_params):
        return self    
  


# In[60]:


def minmax(y):
    return MinMaxScaler().fit_transform(y)


class Minmaxscaler():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X_scaled, **transform_params):
        X_scaled[self.columns] = minmax(X_scaled[self.columns])
        return X_scaled
    
    def fit(self, X_scaled, y=None, **fit_params):
        return self
        


# In[61]:


def label(z):
    return z.apply(lambda x: LabelEncoder().fit_transform(x))


class Labelencoder():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X_scaled, **transform_params):
        X_scaled[self.columns] = label(X_scaled[self.columns])
        return X_scaled
    
    def fit(self, X_scaled, y=None, **fit_params):
        return self  
        


# In[62]:


# Using Pipeline to get scaling and modelling

num_features = list(X_train.select_dtypes(exclude="object"))
num_transformer = Pipeline([("scaler", MinMaxScaler())])

cat_features = list(X_train.select_dtypes(include="object"))
cat_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessing = ColumnTransformer([("first", num_transformer, num_features),
          ("second", cat_transformer, cat_features)
          ])


# In[ ]:





# In[ ]:





# In[63]:


def model_test(model_name): # test model performance for default parameters
    pipeline = Pipeline([("preprocessing", preprocessing),
               ("model", model_name)])
    
    global model_fitted
    model_fitted = pipeline.fit(X_train, y_train)
    train_pred = model_fitted.predict(X_train)
    test_pred = model_fitted.predict(X_test)
    print("Train_set Accuracy: %.2f" % accuracy_score(y_train, train_pred))
    print("-----------------------------------------")
    print("Test_set Accuracy: %.2f" % accuracy_score(y_test, test_pred))
    print("-----------------------------------------")
    print("Test_set Confusion Matrix: \n", confusion_matrix(y_test, test_pred))
    print("-----------------------------------------")
    print("Test_report: \n", classification_report(y_test, test_pred))


# In[64]:


def parameter_search(model_name,model_params): # searching best parameter of models
    pipeline = Pipeline([("preprocessing", preprocessing),
               ("model", model_name)])
    
    best_params = GridSearchCV(pipeline, model_params, 
                               cv = 10, scoring = "accuracy").fit(X_train, y_train).best_params_
    return best_params


# In[65]:


def optimization(model_name, params): # model test and scoring metrics with hyperparameters   
    pipeline = Pipeline([("preprocessing", preprocessing),
               ("model", model_name)])
    global model_tuned
    model_tuned = pipeline.set_params(**params).fit(X_train, y_train)
    train_pred = model_tuned.predict(X_train)
    test_pred = model_tuned.predict(X_test)
    print("Tuned Model Evaluating")
    print("=========================================")
    print("=========================================\n")
    print("Train_set Accuracy: %.2f" % accuracy_score(y_train, train_pred))
    print("-----------------------------------------")
    print("Test_set Accuracy: %.2f" % accuracy_score(y_test, test_pred))
    print("-----------------------------------------")
    print("Test_set Confusion Matrix: \n", confusion_matrix(y_test, test_pred))
    print("-----------------------------------------")
    print("Test_report: \n", classification_report(y_test, test_pred))
    print("-----------------------------------------")
    print("Test_curve: \n")
    
    model_roc_auc = roc_auc_score(y_test, test_pred)

    fpr, tpr, thresholds = roc_curve(y_test, model_tuned.predict_proba(X_test)[:,1])
    plt.figure(figsize = (7,5))
    plt.plot(fpr, tpr, label = "AUC (area = %0.2f)" % model_roc_auc)
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive", fontsize = 12, fontweight = "bold")
    plt.ylabel("True Positive", fontsize = 12, fontweight = "bold")
    plt.title("Tuned Model ROC Curve", fontsize = 11, fontweight = "bold", color = "brown")
    plt.show()
    
    return model_tuned


# In[ ]:





# # Logistic Regression

# In[66]:


from sklearn.linear_model import LogisticRegression


# In[67]:


model_test(LogisticRegression(max_iter = 1000))


# In[68]:


log_params = {"model__C" : [0.1,1,5],
             "model__solver" : ["lbfgs", "liblinear", "sag"]}


# In[69]:


log_best_params = parameter_search(LogisticRegression(max_iter = 1000),log_params)
log_best_params


# In[70]:


log_tuned = optimization(LogisticRegression(),log_best_params)
log_tuned


# In[ ]:





# # KNN

# In[71]:


from sklearn.neighbors import KNeighborsClassifier


# In[72]:


model_test(KNeighborsClassifier())


# In[73]:


knn_params = {"model__n_neighbors" : [5,10,15,20]}


# In[74]:


knn_best_params = parameter_search(KNeighborsClassifier(), knn_params)
knn_best_params


# In[75]:


knn_tuned = optimization(KNeighborsClassifier(), knn_best_params)
knn_tuned


# In[ ]:





# # Random Forest

# In[76]:


from sklearn.ensemble import RandomForestClassifier


# In[77]:


model_test(RandomForestClassifier())


# In[78]:


rf_params = {"model__n_estimators" : [100,500,750],
            "model__max_depth" : [50,100]}


# In[79]:


rf_best_params = parameter_search(RandomForestClassifier(), rf_params)
rf_best_params


# In[80]:


rf_tuned = optimization(RandomForestClassifier(), rf_best_params)
rf_tuned


# In[ ]:





# # XGBoost Classifier

# In[81]:


from xgboost import XGBClassifier


# In[82]:


model_test(XGBClassifier())


# In[83]:


xgb_params = {"model__n_estimators": [100, 500],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.1,0.01]}


# In[84]:


xgb_best_params = parameter_search(XGBClassifier(), xgb_params)
xgb_best_params


# In[85]:


xgb_tuned = optimization(XGBClassifier(), xgb_best_params)
xgb_tuned


# In[ ]:





# In[87]:


# model save

from joblib import dump
dump(rf_tuned, "rf_bank_model.pkl")


