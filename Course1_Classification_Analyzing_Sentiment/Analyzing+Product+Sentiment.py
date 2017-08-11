
# coding: utf-8

# # Predicting sentiment from product reviews
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](/notebooks/Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# In[2]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[3]:

products = graphlab.SFrame('amazon_baby.gl/')


# # Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[4]:

products.head()


# # Build the word count vector for each review

# In[5]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[6]:

products.head()


# In[7]:

graphlab.canvas.set_target('ipynb')


# In[8]:

products['name'].show()


# # Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[9]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[10]:

len(giraffe_reviews)


# In[11]:

giraffe_reviews['rating'].show(view='Categorical')


# # Build a sentiment classifier

# In[12]:

products['rating'].show(view='Categorical')


# ## Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[13]:

# ignore all 3* reviews
products = products[products['rating'] != 3]


# In[14]:

# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[15]:

products.head()


# ## Let's train the sentiment classifier

# In[16]:

train_data,test_data = products.random_split(.8, seed=0)


# In[17]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# # Evaluate the sentiment model

# In[18]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[19]:

sentiment_model.show(view='Evaluation')


# # Applying the learned model to understand sentiment for Giraffe

# In[20]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[21]:

giraffe_reviews.head()


# ## Sort the reviews based on the predicted sentiment and explore

# In[22]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[23]:

giraffe_reviews.head()


# ## Most positive reviews for the giraffe

# In[24]:

giraffe_reviews[0]['review']


# In[25]:

giraffe_reviews[1]['review']


# ## Show most negative reviews for giraffe

# In[26]:

giraffe_reviews[-1]['review']


# In[27]:

giraffe_reviews[-2]['review']


# ## Write a function to count the times of selected words in review

# In[ ]:




# In[28]:

def awesome_count(dict):
    if 'awesome' in dict:
        return dict['awesome']
    else:
        return 0
products['awesome'] = products['word_count'].apply(awesome_count)
def great_count(dict):
    if 'great' in dict:
        return dict['great']
    else:
        return 0
products['great'] = products['word_count'].apply(great_count)
def fantastic_count(dict):
    if 'fantastic' in dict:
        return dict['fantastic']
    else:
        return 0
products['fantastic'] = products['word_count'].apply(fantastic_count)
def amazing_count(dict):
    if 'amazing' in dict:
        return dict['amazing']
    else:
        return 0
products['amazing'] = products['word_count'].apply(amazing_count)
def love_count(dict):
    if 'love' in dict:
        return dict['love']
    else:
        return 0
products['love'] = products['word_count'].apply(love_count)
def horrible_count(dict):
    if 'horrible' in dict:
        return dict['horrible']
    else:
        return 0
products['horrible'] = products['word_count'].apply(horrible_count)
def bad_count(dict):
    if 'bad' in dict:
        return dict['bad']
    else:
        return 0
products['bad'] = products['word_count'].apply(bad_count)
def terrible_count(dict):
    if 'terrible' in dict:
        return dict['terrible']
    else:
        return 0
products['terrible'] = products['word_count'].apply(terrible_count)
def awful_count(dict):
    if 'awful' in dict:
        return dict['awful']
    else:
        return 0
products['awful'] = products['word_count'].apply(awful_count)
def wow_count(dict):
    if 'wow' in dict:
        return dict['wow']
    else:
        return 0
products['wow'] = products['word_count'].apply(wow_count)
def hate_count(dict):
    if 'hate' in dict:
        return dict['hate']
    else:
        return 0
products['hate'] = products['word_count'].apply(hate_count)
products.head()


# ## 1. Count the number of occurence of selected_words on the columns

# In[29]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[30]:

d={}
for i in selected_words:
    d[i]= products[i].sum()
a=min(d, key=d.get)
b=max(d, key=d.get)
print "the least used: "+ a
print "the most used: "+ b



# 

# In[45]:

print products['sentiment'].sum()/float(len(products['name']))


# ## The accuracy of majority class classifier: 0.841123344847

# In[44]:

train_data,test_data = products.random_split(.8, seed=0)
selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features= selected_words,
                                                     validation_set=test_data)


# ## 2. Define the weight of each selected words

# In[33]:

selected_words_model['coefficients'].sort('value').print_rows(12)


# # "terrible" has the most negative, and "love" has the most positive

# ## 3. Comparing the accuracy of different sentiment analysis model

# In[34]:

selected_words_model.evaluate(test_data)


# In[35]:

sentiment_model.evaluate(test_data)


# ## 4.Interpreting the difference in performance between the models:

# In[36]:

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']


# In[37]:

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')


# In[ ]:




# In[38]:

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)


# In[39]:

diaper_champ_reviews.head()


# # The most positive review of sentiment_model has 0.9984 predicted sentiment

# In[46]:

selected_words_model.predict(diaper_champ_reviews[0:10], output_type='probability')


# ## The most positive review of selected_words_model has 0.79694.. predicted sentiment

# In[ ]:



