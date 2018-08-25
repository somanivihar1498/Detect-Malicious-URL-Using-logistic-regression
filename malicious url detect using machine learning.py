
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[4]:


#load url_data
url_Data=pd.read_csv('urldata.csv')


# In[5]:


url_Data.head(10)


# In[9]:


#check missing data
url_Data.isnull().sum().sum()


# In[10]:


#Data vectorization using Tfidfvectorizer
#creating user defined tokenizer(not using default tokenoizer)
#spliting data,removing repetition and "com"
def Custom_Tokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com 
    return total_Tokens


# In[11]:


#labels
y=url_Data['label']


# In[12]:


#url list 
url_list=url_Data['url']


# In[13]:


# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=Custom_Tokens)


# In[14]:


# Store vectors into X variable as XFeatures
X = vectorizer.fit_transform(url_list)


# In[15]:


#spliting data into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[16]:


#training model using logistic regression
logistic=LogisticRegression()
logistic.fit(X_train,y_train)


# In[17]:


#checking accuracy of model using test data
print("Accuracy of model is:",logistic.score(X_test,y_test))


# In[18]:


#checking accuracy of model using train data
print("Accuracy of model is :",logistic.score(X_train,y_train))


# In[25]:


#prediction with model created
X_predict = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe"]


# In[26]:


X_predict = vectorizer.transform(X_predict)
New_predict = logistic.predict(X_predict)


# In[27]:


print(New_predict)


# In[31]:


X_predict1 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]


# In[32]:


X_predict1 = vectorizer.transform(X_predict1)
New_predict1 = logistic.predict(X_predict1)
print(New_predict1)


# In[33]:


#using default tokenizer
vectorizer=TfidfVectorizer()


# In[34]:


y=url_Data['label']


# In[35]:


X=url_Data['url']


# In[41]:


X = vectorizer.fit_transform(url_list)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[42]:


logistic1=LogisticRegression()


# In[43]:


logistic1.fit(X_train,y_train)


# In[44]:


#accuracy of model
print("Accuracy of model using default tokenizer is:",logistic1.score(X_test,y_test))


# In[45]:


print("Accuracy of model using default tokenizer is:",logistic1.score(X_train,y_train))


# In[46]:


X_predict2 = ["www.buyfakebillsonlinee.blogspot.com", 
"www.unitedairlineslogistics.com",
"www.stonehousedelivery.com",
"www.silkroadmeds-onlinepharmacy.com" ]


# In[47]:


X_predict2=vectorizer.transform(X_predict2)


# In[48]:


New_predict2=logistic1.predict(X_predict2)


# In[49]:


print(New_predict2)


# In[53]:


from sklearn.metrics import confusion_matrix


# In[56]:


predicted = logistic1.predict(X_test)
matrix = confusion_matrix(y_test, predicted)


# In[57]:


print(matrix)


# In[58]:


from sklearn.metrics import classification_report


# In[59]:


report = classification_report(y_test, predicted)
print(report)

