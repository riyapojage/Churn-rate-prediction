# In[1]:


import pandas as pd


# In[3]:


df=pd.read_excel(r"C:\Users\91898\Downloads\customer_churn_large_dataset (1).xlsx")
df.head()
df.duplicated().sum()
df=df.drop_duplicates()
# In[4]:


df=df.iloc[:,df.columns!='Name']
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df=pd.get_dummies(df,drop_first=True)
df


# In[16]:


x=df.iloc[:,df.columns!='CustomerID']
x=x.iloc[:,x.columns!='Churn']

x.head()


# In[17]:


y=df['Churn']


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


scaler=StandardScaler()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[23]:


scaler.fit(x_train)


# In[26]:


scaled_x_train=scaler.transform(x_train)
scaled_x_test=scaler.transform(x_test)


# In[28]:


from sklearn.tree import DecisionTreeClassifier


# In[359]:


tree=DecisionTreeClassifier(max_depth=3,min_samples_split=5000)


# In[360]:


from sklearn.ensemble import AdaBoostClassifier


# In[361]:


ada=AdaBoostClassifier(n_estimators=500,estimator=tree)


# In[364]:


ada.fit(scaled_x_train,y_train)


# In[365]:


y_pred=ada.predict(scaled_x_test)


# In[366]:


from sklearn.metrics import accuracy_score


# In[367]:


accuracy_score(y_pred,y_test)


# In[362]:


from sklearn.model_selection import GridSearchCV


# In[370]:


params={"learning_rate":[0.01,0.02,0.03,0.05,0.08,0.1],}


# In[371]:


grid=GridSearchCV(ada,params,n_jobs=-1,scoring='accuracy')


# In[372]:


grid.fit(scaled_x_train,y_train)


# In[373]:


grid.best_params_


# In[376]:


model=grid.best_estimator_


# In[375]:


model.fit(scaled_x_train,y_train)


# In[377]:


y_pred=model.predict(scaled_x_test)


# In[378]:


accuracy_score(y_pred,y_test)


# In[381]:


from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score


# In[384]:


confusion_matrix(y_pred,y_test)


# In[385]:


f1_score(y_pred,y_test)


# In[386]:


recall_score(y_pred,y_test)


# In[387]:


precision_score(y_pred,y_test)


# In[390]:


import pickle


# In[394]:


pickle.dump(model,open('model.pkl','wb'))


# In[ ]:
model=pickle.load(open('model.pkl','rb'))
