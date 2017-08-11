
# coding: utf-8

# # Fire Up Graph Lab Create
# 

# In[1]:

import graphlab
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Load some house sale data

# In[2]:

sales= graphlab.SFrame('home_data.gl/')


# In[3]:

sales


# # Exploring the data for housing

# In[4]:

graphlab.canvas.set_target('ipynb')
sales.show(view="Scatter Plot", x="sqft_living", y="price")


# # Create the simple regression model of sqoft_living to price

# In[5]:

train_data, test_data = sales.random_split(0.8, seed=0)


# ## Build the regression model

# In[6]:

sqft_model= graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'])


# # Evaluate the simple model

# In[7]:

print test_data['price'].mean()


# In[8]:

print sqft_model.evaluate(test_data)


# # Let's show what our predictions look like

# In[9]:

import matplotlib.pyplot as plt


# In[10]:

get_ipython().magic(u'matplotlib inline')


# In[11]:

plt.plot(test_data['sqft_living'], test_data['price'],'.',test_data['sqft_living'], sqft_model.predict(test_data),'-')


# In[12]:

sqft_model.get('coefficients')


# In[13]:

my_features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# In[14]:

sales[my_features].show()


# In[15]:

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# In[16]:

my_feature_model=graphlab.linear_regression.create(train_data, target='price', features=my_features,validation_set=None)


# In[17]:

print my_features


# In[18]:

print sqft_model.evaluate(test_data)
print my_feature_model.evaluate(test_data)


# # Applied learned models to predict prices of 3 houses

# In[19]:

house1 = sales[sales['id']=='5309101200']


# In[20]:

house1


# In[21]:

print house1['price']


# In[22]:

print sqft_model.predict(house1)


# In[23]:

print my_feature_model.predict(house1)


# # Prediction fro the second house , a fancy house

# In[24]:

house2 = sales[sales['id']=='1925069082']


# In[25]:

house2


# In[26]:

print house2['price']


# <img src="https://ssl.cdn-redfin.com/photo/1/bigphoto/302/734302_0.jpg">

# In[27]:

print sqft_model.predict(house2)


# In[28]:

print my_feature_model.predict(house2)


# In[29]:

bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Bill_gates%27_house.jpg/2560px-Bill_gates%27_house.jpg">

# In[30]:

print my_feature_model.predict(graphlab.SFrame(bill_gates))


# # Compute the average price at 98039 zipcode

# In[31]:


average = sales[sales['zipcode']=='98039']
print average


# In[32]:

print average['price'].mean()


# # 2. Filtering data:

# In[33]:

selected_house = sales[(sales['sqft_living']>2000) & (sales['sqft_living']<=4000)].num_rows()


# In[34]:

total = sales[sales['sqft_living']].num_rows()


# In[35]:

print total, selected_house


# In[36]:

print selected_house/ float(total)


# # 3. Computing the RMSE

# In[37]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[38]:

advanced_features_model=graphlab.linear_regression.create(train_data, target='price', features=advanced_features,validation_set=None)


# In[39]:

print advanced_features_model.evaluate(test_data)


# In[40]:

print 179542.43331269105 - 156831.11680191013


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



