#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt


# In[5]:


titanic_train=pd.read_csv('titanic_train.csv')
titanic_test=pd.read_csv('titanic_test.csv')


# In[6]:


titanic_train.head()


# In[7]:


titanic_train.shape


# In[8]:


titanic_train['Survived'].value_counts()


# In[14]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color=["r","g"])
plt.show()


# In[16]:


titanic_train['Pclass'].value_counts


# In[19]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Pclass'].value_counts().keys()),list(titanic_train['Pclass'].value_counts()),color=["blue","green","orange"])
plt.show()


# In[21]:


titanic_train['Sex'].value_counts()


# In[22]:


plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color=["r","g"])
plt.show()


# In[23]:


plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'])
plt.title("Distribution of age")
plt.xlabel("Age")

plt.show()


# In[24]:


sum(titanic_train['Survived'].isnull())


# In[25]:


sum(titanic_train['Age'].isnull())


# In[26]:


titanic_train=titanic_train.dropna()


# In[27]:


#building model


# In[28]:


sum(titanic_train['Survived'].isnull())


# In[29]:


sum(titanic_train['Age'].isnull())


# In[30]:


x_train=titanic_train[['Age']]
y_train=titanic_train[['Survived']] 


# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


dtc= DecisionTreeClassifier()


# In[33]:


dtc.fit(x_train,y_train)


# In[34]:


#predicting value


# In[ ]:





# In[36]:


sum(titanic_test['Age'].isnull())


# In[39]:


titanic_test=titanic_test.dropna()


# In[40]:


sum(titanic_test['Age'].isnull())


# In[47]:


x_test=titanic_test[['Age']]


# In[48]:


y_pred=dtc.predict(x_test)


# In[49]:


y_pred


# In[ ]:




