#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Introduction
#The story of Enron is a story of a company that reached immense heights to deepest lows in no time. 
#Enron Corp. was one of the biggest firms in the United States and was delivering splendid performance 
#on wall street. However, the company witnessed a sudden drop in prices and declared bankruptcy. 
#How one of the most powerful businesses in the US, and the world, disintegrated overnight is still
#a puzzle to many.
#The Enron leadership was involved in one of the biggest frauds and this particular fraud has been 
#an area of interest for many researchers and ML practitioners.
#In this case study, we have a subset of 50 senior officials. The idea is to build a network from the 
#emails, sent and received by those senior officials, to better understand the connections and highlight 
#the important nodes in this group.


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from decorator import decorator
import networkx as nx
from networkx.utils import create_random_state, create_py_random_state


# In[4]:


data = pd.read_csv('/Users/yutaoyan/Desktop/Enron/EmailEnron.csv')


# In[6]:


data.shape


# In[7]:


data.head()


# In[8]:


G = nx.Graph()


# In[9]:


G = nx.from_pandas_edgelist(data, 'From', 'To')

plt.figure(figsize = (10, 10))

options = {
    "node_color": "black",
    "node_size": 10,
    "linewidths": 0.5,
    "width": 0.1,
}

nx.draw(G, with_labels = True, **options)

plt.show()


# In[10]:


plt.figure(figsize = (10, 10))

nx.draw_shell(G, with_labels = True, **options)


# In[11]:


plt.figure(figsize = (10, 10))

# With the default parameters
nx.draw_spring(G, with_labels = True)


# In[12]:


#Out of the 80 nodes in the dataset, 1 appears to be the most important node as it is connected with all 
#the other nodes. We can interpret this official, perhaps as the CEO.
#Other important nodes are also highlighted in the visualization - 56, 54, 74, 53, 50. The circular 
#visualization is a better visualization approach to highlight the important nodes.
#There are internal team structures that appear from the visualization but are not very clear as to which 
#nodes are part of which teams.


# In[13]:


# Let us quickly look at the degree of the nodes
for i in G.degree():
    print(i)


# In[14]:


#Centrality Measures


# In[15]:


deg_cen = nx.degree_centrality(G)

eig_cen = nx.eigenvector_centrality(G)

clo_cen = nx.closeness_centrality(G)

bet_cen = nx.betweenness_centrality(G)


# In[16]:


#a. Degree Centrality
temp = {}

for w in sorted(deg_cen, key = deg_cen.get, reverse = True):
    temp[w] = deg_cen[w]

print("Sorted Importance of nodes in terms of deg_cen for Phase {} is {}".format(w + 1, list(temp.keys())[:5]))

print()


# In[17]:


# Let us color these nodes and visualize the network again

color = []

for node in G:
    
    if (node == 1 or node == 56 or node == 74 or node==53 or node==54):
        color.append('red')
    
    else:
        color.append('black')

plt.figure(figsize = (10, 10))

nx.draw(G, node_color = color, with_labels = True)


# In[18]:


#b. Eigenvector Centrality
temp = {}

for w in sorted(eig_cen, key = eig_cen.get, reverse = True):
    temp[w] = eig_cen[w]

print("Sorted Importance of nodes in terms of eig_cen for Phase {} is {}".format(w + 1, list(temp.keys())[:5]))

print()


# In[19]:


# Let us color these nodes and visualize the network again

color = []

for node in G:
    
    if (node == 1 or node == 56  or node == 74 or node==53 or node==54):
        color.append('red')
    
    else:
        color.append('black')

plt.figure(figsize = (10, 10))

nx.draw(G, node_color = color, with_labels = True)


# In[20]:


#c. Betweenness Centrality


# In[21]:


temp = {}

for w in sorted(bet_cen, key = bet_cen.get, reverse = True):
    temp[w] = bet_cen[w]

print("Sorted Importance of nodes in terms of bet_cen is {}".format(list(temp.keys())[:5]))

print()


# In[22]:


color = []

for node in G:
    
    if (node == 1 or node == 56  or node == 54 or node==27 or node==74):
        color.append('red')
    
    else:
        color.append('black')

plt.figure(figsize = (10, 10))

nx.draw(G, node_color = color, with_labels = True)


# In[23]:


#d. Closeness Centrality


# In[24]:


temp = {}

for w in sorted(clo_cen, key = clo_cen.get, reverse = True):
    temp[w] = clo_cen[w]

print("Sorted Importance of nodes in terms of clo_cen is {}".format(list(temp.keys())[:5]))

print()


# In[25]:


color = []
for node in G:
    
    if (node == 1 or node == 56  or node == 53 or node==54 or node==27):
        color.append('red')
    
    else:
        color.append('black')

plt.figure(figsize = (10, 10))

nx.draw(G, node_color = color, with_labels = True)


# In[ ]:


#Conclusion
#We figured out the connections in the organization by visualizing the network.
#We also found various centrality measures and figured out the important nodes for each centrality measure. 
#The importance of these nodes can be further explained by the definitions of the centralities they correspond to.
#We also identified the CEO node, i.e., Node 1. Nodes 56 and 54 are the other two nodes considered important by 
#each centrality measure.
#We could figure out that there were internal team structures, but the connections were not very clear.

