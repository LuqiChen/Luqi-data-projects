#!/usr/bin/env python
# coding: utf-8

# In[1]:


from twitter import *


# In[2]:


import sys
sys.path.append(".")


# In[3]:


import config


# In[4]:


twitter = Twitter(auth = OAuth(config.access_key,
                  config.access_secret,
                  config.consumer_key,
                  config.consumer_secret))


# In[5]:


user = "nytimes"


# In[28]:


results = twitter.statuses.user_timeline(screen_name = user,count=5,tweet_mode='extended')


# In[31]:


for status in results:
    print("(%s) %s" % (status["created_at"], status["retweet_count"]), status["full_text"].encode("ascii", "ignore"))


# In[39]:


data=[]
for status in results:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data.append([time,retweet_count,favorite_count,text])


# In[40]:


print(data)


# In[43]:


import pandas as pd
df = pd.DataFrame(data)
df.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
print(df)


# In[44]:


df.to_csv("5data.csv")


# In[9]:


results1 = twitter.statuses.user_timeline(screen_name = user,count=1)


# In[30]:


print(results[0])


# In[46]:


result_9_9 = twitter.statuses.user_timeline(screen_name = user,count= 800,tweet_mode='extended')
data_9_9=[]
for status in result_9_9:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_9.append([time,retweet_count,favorite_count,text])
    
df_9_9 = pd.DataFrame(data_9_9)
df_9_9.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_9.to_csv("df_9_9.csv")


# In[6]:


result_9_10 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_10=[]
for status in result_9_10:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_10.append([time,retweet_count,favorite_count,text])
    
df_9_10 = pd.DataFrame(data_9_10)
df_9_10.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_10.to_csv("df_9_10.csv")


# In[9]:


result_9_11 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_11=[]
for status in result_9_11:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_11.append([time,retweet_count,favorite_count,text])
    
df_9_11 = pd.DataFrame(data_9_11)
df_9_11.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_11.to_csv("df_9_11.csv")


# In[7]:


result_9_12 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_12=[]
for status in result_9_12:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_12.append([time,retweet_count,favorite_count,text])
    
df_9_12 = pd.DataFrame(data_9_12)
df_9_12.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_12.to_csv("df_9_12.csv")


# In[8]:


result_9_13 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_13=[]
for status in result_9_13:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_13.append([time,retweet_count,favorite_count,text])
    
df_9_13 = pd.DataFrame(data_9_13)
df_9_13.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_13.to_csv("df_9_13.csv")


# In[8]:


result_9_15 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_15=[]
for status in result_9_15:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_15.append([time,retweet_count,favorite_count,text])
    
df_9_15 = pd.DataFrame(data_9_15)
df_9_15.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_15.to_csv("df_9_15.csv")


# In[9]:


result_9_16 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_16=[]
for status in result_9_16:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_16.append([time,retweet_count,favorite_count,text])
    
df_9_16 = pd.DataFrame(data_9_16)
df_9_16.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_16.to_csv("df_9_16.csv")


# In[10]:


result_9_17 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_17=[]
for status in result_9_17:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_17.append([time,retweet_count,favorite_count,text])
    
df_9_17 = pd.DataFrame(data_9_17)
df_9_17.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_17.to_csv("df_9_17.csv")


# In[11]:


result_9_18 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_18=[]
for status in result_9_18:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_18.append([time,retweet_count,favorite_count,text])
    
df_9_18 = pd.DataFrame(data_9_18)
df_9_18.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_18.to_csv("df_9_18.csv")


# In[12]:


result_9_20 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_20=[]
for status in result_9_20:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_20.append([time,retweet_count,favorite_count,text])
    
df_9_20 = pd.DataFrame(data_9_20)
df_9_20.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_20.to_csv("df_9_20.csv")


# In[7]:


result_9_21 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_21=[]
for status in result_9_21:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_21.append([time,retweet_count,favorite_count,text])
    
df_9_21 = pd.DataFrame(data_9_21)
df_9_21.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_21.to_csv("df_9_21.csv")


# In[8]:


result_9_22 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_22=[]
for status in result_9_22:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_22.append([time,retweet_count,favorite_count,text])
    
df_9_22 = pd.DataFrame(data_9_22)
df_9_22.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_22.to_csv("df_9_22.csv")


# In[9]:


result_9_23 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_23=[]
for status in result_9_23:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_23.append([time,retweet_count,favorite_count,text])
    
df_9_23 = pd.DataFrame(data_9_23)
df_9_23.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_23.to_csv("df_9_23.csv")


# In[10]:


result_9_24 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_24=[]
for status in result_9_24:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_24.append([time,retweet_count,favorite_count,text])
    
df_9_24 = pd.DataFrame(data_9_24)
df_9_24.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_24.to_csv("df_9_24.csv")


# In[11]:


result_9_25 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_25=[]
for status in result_9_25:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_25.append([time,retweet_count,favorite_count,text])
    
df_9_25 = pd.DataFrame(data_9_25)
df_9_25.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_25.to_csv("df_9_25.csv")


# In[12]:


result_9_26 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_26=[]
for status in result_9_26:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_26.append([time,retweet_count,favorite_count,text])
    
df_9_26 = pd.DataFrame(data_9_26)
df_9_26.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_26.to_csv("df_9_26.csv")


# In[13]:


result_9_28 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_28=[]
for status in result_9_28:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_28.append([time,retweet_count,favorite_count,text])
    
df_9_28 = pd.DataFrame(data_9_28)
df_9_28.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_28.to_csv("df_9_28.csv")


# In[14]:


result_9_29 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_29=[]
for status in result_9_29:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_29.append([time,retweet_count,favorite_count,text])
    
df_9_29 = pd.DataFrame(data_9_29)
df_9_29.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_29.to_csv("df_9_29.csv")


# In[15]:


result_9_30 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_9_30=[]
for status in result_9_30:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_9_30.append([time,retweet_count,favorite_count,text])
    
df_9_30 = pd.DataFrame(data_9_30)
df_9_30.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_9_30.to_csv("df_9_30.csv")


# In[16]:


result_10_01 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_01=[]
for status in result_10_01:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_01.append([time,retweet_count,favorite_count,text])
    
df_10_01 = pd.DataFrame(data_10_01)
df_10_01.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_01.to_csv("df_10_01.csv")


# In[17]:


result_10_02 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_02=[]
for status in result_10_02:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_02.append([time,retweet_count,favorite_count,text])
    
df_10_02 = pd.DataFrame(data_10_02)
df_10_02.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_02.to_csv("df_10_02.csv")


# In[18]:


result_10_03 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_03=[]
for status in result_10_03:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_03.append([time,retweet_count,favorite_count,text])
    
df_10_03 = pd.DataFrame(data_10_03)
df_10_03.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_03.to_csv("df_10_03.csv")


# In[19]:


result_10_04 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_04=[]
for status in result_10_04:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_04.append([time,retweet_count,favorite_count,text])
    
df_10_04 = pd.DataFrame(data_10_04)
df_10_04.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_04.to_csv("df_10_04.csv")


# In[20]:


result_10_05 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_05=[]
for status in result_10_05:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_05.append([time,retweet_count,favorite_count,text])
    
df_10_05 = pd.DataFrame(data_10_05)
df_10_05.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_05.to_csv("df_10_05.csv")


# In[21]:


result_10_06 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_06=[]
for status in result_10_06:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_06.append([time,retweet_count,favorite_count,text])
    
df_10_06 = pd.DataFrame(data_10_06)
df_10_06.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_06.to_csv("df_10_06.csv")


# In[7]:


result_10_08 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_08=[]
for status in result_10_08:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_08.append([time,retweet_count,favorite_count,text])
    
df_10_08 = pd.DataFrame(data_10_08)
df_10_08.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_08.to_csv("df_10_08.csv")


# In[8]:


result_10_09 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_09=[]
for status in result_10_09:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_09.append([time,retweet_count,favorite_count,text])
    
df_10_09 = pd.DataFrame(data_10_09)
df_10_09.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_09.to_csv("df_10_09.csv")


# In[9]:


result_10_10 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_10=[]
for status in result_10_10:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_10.append([time,retweet_count,favorite_count,text])
    
df_10_10 = pd.DataFrame(data_10_10)
df_10_10.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_10.to_csv("df_10_10.csv")


# In[10]:


result_10_11 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_11=[]
for status in result_10_11:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_11.append([time,retweet_count,favorite_count,text])
    
df_10_11 = pd.DataFrame(data_10_11)
df_10_11.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_11.to_csv("df_10_11.csv")


# In[11]:


result_10_13 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_13=[]
for status in result_10_13:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_13.append([time,retweet_count,favorite_count,text])
    
df_10_13 = pd.DataFrame(data_10_13)
df_10_13.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_13.to_csv("df_10_13.csv")


# In[12]:


result_10_14 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_14=[]
for status in result_10_14:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_14.append([time,retweet_count,favorite_count,text])
    
df_10_14 = pd.DataFrame(data_10_14)
df_10_14.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_14.to_csv("df_10_14.csv")


# In[13]:


result_10_18 = twitter.statuses.user_timeline(screen_name = user,count= 500,tweet_mode='extended')
data_10_18=[]
for status in result_10_18:
    time = status["created_at"]
    retweet_count = status["retweet_count"]
    favorite_count=status["favorite_count"]
    text= status["full_text"].encode("ascii", "ignore")
    data_10_18.append([time,retweet_count,favorite_count,text])
    
df_10_18 = pd.DataFrame(data_10_18)
df_10_18.columns = ['created_at', 'retweet_count', 'favorite_count', 'text']
df_10_18.to_csv("df_10_18.csv")

