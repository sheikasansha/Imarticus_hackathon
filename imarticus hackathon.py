#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from impyute.imputation.cs import mice

# Visualisation
import matplotlib as mpl #visualization library in Python for 2D plots of arrays
                         #generate plots, histograms, 
                         #power spectra, bar charts, 
                         #errorcharts, scatterplots, etc.

import matplotlib.pyplot as plt #functions provides a MATLAB-like plotting framework.                                
                                #creates a figure, creates a plotting area in a figure, 
                                #plots some lines in a plotting area, decorates the plot with labels, etc.

import matplotlib.pylab as pylab #pylab combines pyplot with numpy into a single namespace
import seaborn  as sns # seaborn is for statistical visualisation
import missingno as msno # ibrary offers a very nice way to visualize the distribution of NaN values.


# In[2]:


#from impyute.imputation.cs import mice


# In[3]:


#from sklearn.preprocessing import Imputer


# In[4]:


# Loading the training data set
train_data=pd.read_csv("F:\\Imarticus\\Imarticus Hackathon\\TRAINING.csv")
test_data=pd.read_csv("F:\\Imarticus\\Imarticus Hackathon\\TEST.csv")


# In[5]:


# Creating a copy of dataframe
df = train_data.copy()
df_t=test_data.copy()


# In[6]:


# analysing the data types
print(df.info())
print(df_t.info())


# In[7]:


# statistical analyse of the variable
print(df.describe())
print(df_t.describe())


# In[8]:


# shape of the dataframe
print(df.shape)
print(df_t.shape)
# so there are 7000 observation and 14 variables in train and 3299 observation and 13 variables


# In[9]:


# veiwing the column header
print(df.columns)
print(df_t.columns)


# From the problem statement, we can coonclude that Grade is the Target Variable and we don't need 'id', so we can remove 'id' or we can set 'id' as index

# In[10]:


# setting id as indexes
df.set_index(["id"],inplace=True)
df_t.set_index(["id"],inplace=True)
print(df.head())
print(df_t.head())


# # Exploratory Data Analysis

# Univariate Analysis

# In[11]:


# Analysing continous variable
print(df.describe())
print(df_t.describe())


# here, we can see some missing values in 'Troom', 'Nbedrooms','Nbwashrooms','Twashrooms','Roof(Area)','Lawn(Area)','API'

# In[12]:


# Analysing categorical variable
print(df.describe(include=[object]))
print(df_t.describe(include=[object]))


# from this we can say that 'roof' column has missing values

# In[13]:


print(df.roof.value_counts())
print(df.Grade.value_counts())
print(df.EXPECTED.value_counts())


# In[14]:


print(df_t.roof.value_counts())
print(df_t.EXPECTED.value_counts())


# Here in "roof" there are 4 classes but its a binomial clases. need to change into 2 classes and "EXPECTED" is continous variable. need to change integer

# In[15]:


print(df.roof.value_counts())
print(df_t.roof.value_counts())


# In[16]:


df.roof = df.roof.replace("no","NO")
df.roof = df.roof.replace("yes","YES")
df_t.roof = df_t.roof.replace("no","NO")
df_t.roof = df_t.roof.replace("yes","YES")


# In[17]:


print(df.roof.value_counts())
print(df_t.roof.value_counts())


# In[18]:


# Removing the $ symbol
df["EXPECTED"] = df["EXPECTED"].str.strip("$")
df_t["EXPECTED"] = df_t["EXPECTED"].str.strip("$")


# In[19]:


print(df["EXPECTED"].head())
print(df_t["EXPECTED"].head())


# In[20]:


# converting datatype into int
df["EXPECTED"] = df["EXPECTED"].astype("int64")
df_t["EXPECTED"] = df_t["EXPECTED"].astype("int64")
print(df.EXPECTED.dtype)
print(df_t.EXPECTED.dtype)


# Encoding the category varible

# In[21]:


mapping = {"A":1,"B":2,"C":3,"D":4,"E":5}
mapping1 = {"YES":1,"NO":0}

df["grade_score"] = df["Grade"].replace(mapping)
df["roof_grade"] = df["roof"].replace(mapping1)

df_t["roof_grade"] = df_t["roof"].replace(mapping1)


# In[22]:


# converting into categorical datatype
df["grade_score"] = df["grade_score"].astype("category")
print(df.grade_score.dtype)
df["roof_grade"] = df["roof_grade"].astype("Int64")
df.roof_grade = df.roof_grade.astype("category")
print(df.roof_grade.dtype)

df_t["roof_grade"] = df_t["roof_grade"].astype("Int64")
df_t.roof_grade = df_t.roof_grade.astype("category")
print(df.roof_grade.dtype)


# In[ ]:





# In[23]:


df.info()


# In[24]:


df.head()


# In[25]:


# dropping unencoded features
df.drop(["roof", "Grade"], axis=1, inplace = True)
df_t.drop(["roof"], axis=1, inplace = True)


# In[26]:


print(df.head())
print(df_t.head())


# In[27]:


print(df.ANB.value_counts())
print(df_t.ANB.value_counts())


# when we check the 'ANB' feature, it shows only 6 unique features and from the project description we can assume that this feature is a nominal feature with 6 classes. so we convert this imto category variable

# In[28]:


# df.ANB = df.ANB.astype("category")
# print(df.ANB.dtype)
# df_t.ANB = df_t.ANB.astype("category")
# print(df_t.ANB.dtype)


# Performing Onehot encoding

# In[29]:


# df_dummies = pd.get_dummies(df.ANB, prefix = "ANB", prefix_sep = "_")


# In[30]:


# df_d = pd.concat([df,df_dummies], axis = 1)
# df_d.shape


# In[31]:


# df_d.drop(["ANB"], axis = 1, inplace = True)


# In[32]:


# df_d.shape


# In[33]:


df.head()


# Missing value treatment

# In[34]:


print(df.isna().sum())
print(df_t.isna().sum())


# In[35]:


df.shape


# In[36]:


col_mis = list(df.columns[df.isna().any()])
print(col_mis)

col_mis_t = list(df_t.columns[df_t.isna().any()])
print(col_mis_t)


# In[37]:


# Calculating the missing values for training set
miss_df=pd.DataFrame(columns=["col","miss","perc","uniq"])
miss_df_t=pd.DataFrame(columns=["col","miss","perc","uniq"])
for i in col_mis:
    miss=df[i].isna().sum()
    perc=miss/len(df)*100
    uniq=df[i].value_counts().count()
    miss_df=miss_df.append({"col":i,"miss":miss,"perc":perc,"uniq":uniq},ignore_index=True)


# In[38]:


# calculating the missing values for test set
miss_df_t=pd.DataFrame(columns=["col","miss","perc","uniq"])
for j in col_mis_t:
    miss_t=df_t[j].isna().sum()
    perc_t=miss_t/len(df)*100
    uniq_t=df_t[j].value_counts().count()
    miss_df_t=miss_df_t.append({"col":j,"miss":miss_t,"perc":perc_t,"uniq":uniq_t},ignore_index=True)


# In[39]:


miss_df


# In[40]:


miss_df_t


# In[41]:


df.dropna(subset = ['Troom', 'Nbedrooms', 'Nbwashrooms','Twashrooms','Lawn(Area)','API'],inplace=True)


# In[42]:


df_t.dropna(subset=['Troom', 'Nbedrooms', 'Nbwashrooms','Lawn(Area)','API'],inplace=True)


# In[43]:


df.shape


# In[44]:


df_t.shape


# In[45]:


msno.matrix(df[col_mis],figsize=(5,5))


# In[46]:


df_clean = df.drop(["roof_grade","Roof(Area)"], axis = 1)
df_clean.shape


# In[47]:


df_t_clean=df_t.drop(["roof_grade","Roof(Area)"], axis=1)
df_t_clean.shape


# In[48]:


# Visualising the univariate
df.hist(figsize = (20,20),bins=30) # using matplot


# In[49]:


df["grade_score"].hist(figsize = (5,5),bins=30)


# In[50]:


df["roof_grade"].hist(figsize = (5,5),bins=30)


# # Feature Selection

# Univariate Selection

# In[51]:


# importing packages for univariate selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[52]:


x = df_clean.drop(["grade_score"], axis=1)
y = df_clean.grade_score
print(x.shape, y.shape)


# In[53]:


x_t=df_t_clean
x_t.shape


# In[54]:


#apply SelectKBest class to extract top 10 best features
bf = SelectKBest(score_func=chi2, k=10)
bf_fit = bf.fit(x,y)
dfscores = pd.DataFrame(bf_fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featurescores=pd.concat([dfcolumns,dfscores],axis=1)
featurescores.columns=["spec","score"]


# In[55]:


featurescores.sort_values(by=["score"],ascending=False)


# In[137]:


x_ms=x.drop(["ANB","Lawn(Area)"], axis=1)
x_t_ms=x_t.drop(["ANB","Lawn(Area)","API","Troom"], axis=1)


# In[138]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_ms,y,test_size=0.3,random_state = 0)


# In[139]:


print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)


# In[140]:


#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[141]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
classifier = LogisticRegression(random_state = 0,multi_class='multinomial',solver='newton-cg',max_iter=500)
model = classifier.fit(xtrain,ytrain) 
y_pred=model.predict(xtest)
print(accuracy_score(ytest,y_pred))
print(model.score(xtest,ytest))
# print(f1_score(ytest,y_pred))
# print(precision_score(ytest,y_pred))
# print(recall_score(ytest,y_pred))


# In[142]:


from sklearn.naive_bayes import MultinomialNB 


# In[143]:


nv=MultinomialNB()
model_nv = nv.fit(xtrain,ytrain) 
y_pred_nv=model_nv.predict(xtest)
print(accuracy_score(ytest,y_pred_nv))


# In[144]:


from sklearn.tree import DecisionTreeClassifier


# In[145]:


dt=DecisionTreeClassifier()
model_dt = dt.fit(xtrain,ytrain) 
y_pred_dt=model_dt.predict(xtest)
print(accuracy_score(ytest,y_pred_dt))

