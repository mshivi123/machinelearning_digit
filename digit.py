
# coding: utf-8

# In[60]:


from sklearn import datasets


# In[62]:


mydata=datasets.load_digits()


# In[63]:


mydata.keys()


# In[64]:


mydata.data


# In[65]:


mydata.target


# In[66]:


mydata.images


# In[67]:


mydata.data.shape


# In[68]:


mydata.images.shape


# In[69]:


import numpy as np


# In[70]:


a=np.array([1.,2.,2.])


# In[71]:


a.dtype


# In[72]:


mydata.target[-1]


# In[73]:


mydata.data[-1]


# In[76]:


import matplotlib.pyplot as plt


# In[77]:


myimage=mydata.images[-1]


# In[78]:


myimage


# In[79]:


plt.matshow(myimage)


# In[80]:


myimage.dtype


# In[81]:


x_input=mydata.data


# In[82]:


y_output=mydata.target


# In[83]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x_input,y_output)


# In[87]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[89]:


from sklearn.neighbors import KNeighborsClassifier


# In[91]:


mygau=GaussianNB()
mymul=MultinomialNB()
myber=BernoulliNB()
myknn=KNeighborsClassifier


# In[92]:


mymodel=mygau.fit(x_train,y_train)


# In[93]:


ypguss=mymodel.predict(x_test)


# In[94]:


from sklearn import metrics


# In[95]:


acc_gau=metrics.accuracy_score(y_test,ypguss)


# In[97]:


print("accuracy{}".format(acc_gau))


# In[98]:


from sklearn.svm import SVC


# In[99]:


myobj=SVC(gamma=0.001)


# In[100]:


mymodel=myobj.fit(x_train,y_train)


# In[101]:


from sklearn import metrics


# In[102]:


yp=mymodel.predict(x_test)


# In[104]:


metrics.accuracy_score(y_test,yp)


# In[106]:


metrics.mean_squared_error(y_test,yp)


# In[107]:


mymodel.predict([mydigitdata.data[-1]])

