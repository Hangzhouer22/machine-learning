#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Background 
#Game of Thrones is a wildly popular television series by HBO, based on the (also) wildly popular 
#book series "A Song of Ice and Fire" by George R.R. Martin. In this case study, we will analyze 
#the co-occurrence network of the characters in the Game of Thrones books.

#The dataset is publicly available for the 5 books.

#Note: Here, two characters are considered to co-occur if their names appear in the vicinity of 15 words 
#from one another in the books.


# In[ ]:


#objectives
#Load all the raw datasets and perform descriptive analysis
#Run Network Analysis Algorithms on individual books (and combined)
#Calculate the different centralities measures and provide inference
#Create Network Graphs using Plotly
#Run Louvain Community Detection and find out different groups/communities in the data


# In[1]:


import warnings
warnings.filterwarnings('ignore')     # to avoid warning messages


# In[ ]:


#Installing the necessary libraries


# In[2]:


get_ipython().system('pip install plotly ')
get_ipython().system('pip install community ')
get_ipython().system('pip install python-louvain')
get_ipython().system('pip install colorlover')


# In[3]:


#importing the packages


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import networkx as nx

from decorator import decorator

from networkx.utils import create_random_state, create_py_random_state

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

# Remove scientific notations and display numbers with 2 decimal points instead
pd.options.display.float_format = '{:,.2f}'.format        

# Update default background style of plots
sns.set_style(style='darkgrid')

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly

import plotly.express as px
init_notebook_mode(connected=True)


# In[5]:


#loading the CSV data


# In[9]:


os.listdir("/Users/yutaoyan/Desktop/Game_of_throne/")


# In[10]:


book1 = pd.read_csv("/Users/yutaoyan/Desktop/Game_of_throne/book1.csv")


# In[11]:


book1.head()


# In[ ]:


#This is an example of an Undirected Graph. Undirected graphs have edges that do not have a direction.

#The edges indicate a two-way relationship, such that each edge can be traversed in both directions.


# In[12]:


book2 = pd.read_csv("/Users/yutaoyan/Desktop/Game_of_throne/book2.csv")

book3 = pd.read_csv("/Users/yutaoyan/Desktop/Game_of_throne/book3.csv")

book4 = pd.read_csv("/Users/yutaoyan/Desktop/Game_of_throne/book4.csv")

book5 = pd.read_csv("/Users/yutaoyan/Desktop/Game_of_throne/book5.csv")


# In[13]:


books = [book1, book2, book3, book4, book5]

books_combined = pd.DataFrame()

for book in books:
    books_combined = pd.concat([books_combined, book])

# Grouping the data by Person 2 and Person 1 to avoid multiple entries with the same characters 
books_combined = books_combined.groupby(["Person 2", "Person 1"], as_index = False)["weight"].sum()


# In[14]:


books_combined.info()


# In[15]:


books_combined.describe()


# In[ ]:


#There are 2823 edges in total, or 2823 co-occurrences of characters.
#The minimum weight is 3 (meaning every co-occurrence pair has been observed at least thrice), 
#and the maximum weight is 334.
#The mean weight is 11.56, meaning that on average, two co-occurring characters are mentioned 
#around 12 times together. The median of 5 also implies that it is the maximum weight which is 
#more likely the outlier, which is also affirmed by the fact that 75% of the weight values are 11 or lower.


# In[16]:


books_combined[books_combined["weight"] == 334]


# In[ ]:


#The maximum number of 334 connections is shown below to be between Robert Baratheon and Eddard
#Stark, who as Game of Thrones aficionados will know, were pivotal co-characters in the first book.


# In[ ]:


#Creating a Graph Network (for each book as well as all books combined)


# In[17]:


G1 = nx.from_pandas_edgelist(book1, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G2 = nx.from_pandas_edgelist(book2, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G3 = nx.from_pandas_edgelist(book3, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G4 = nx.from_pandas_edgelist(book4, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G5 = nx.from_pandas_edgelist(book5, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G = nx.from_pandas_edgelist(books_combined, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())


# In[ ]:


#Number of nodes and edges across all books


# In[18]:


nx.info(G)


# In[ ]:


#Creating functions to calculate the number of unique connections per character, 
#Degree Centrality, Eigenvector Centrality, and Betweenness Centrality


# In[19]:


# The number of unique connections

def numUniqueConnec(G):
    numUniqueConnection = list(G.degree())
    
    numUniqueConnection = sorted(numUniqueConnection, key = lambda x:x[1], reverse = True)
    
    numUniqueConnection = pd.DataFrame.from_dict(numUniqueConnection)
    
    numUniqueConnection.columns = (["Character", "NumberOfUniqueHCPConnections"])
    
    return numUniqueConnection


# In[20]:


numUniqueConnec(G)


# In[ ]:


#Observation:

#Tyrion Lannister is the character with the highest number of unique connections, followed by Jon
#Snow and Jaime Lannister.


# In[21]:


# Degree Centrality

def deg_central(G):
    deg_centrality = nx.degree_centrality(G)
    
    deg_centrality_sort = sorted(deg_centrality.items(), key = lambda x:x[1], reverse = True)
    
    deg_centrality_sort = pd.DataFrame.from_dict(deg_centrality_sort)
    
    deg_centrality_sort.columns = (["Character", "Degree Centrality"])
    
    return deg_centrality_sort


# In[22]:


deg_centrality_sort = deg_central(G)
deg_central(G)


# In[ ]:


#Observation:
#Tyrion Lannister is the character with the highest Degree Centrality, followed 
#by Jon Snow and Jaime Lannister.
#The higher the number of connections, the higher the Degree Centrality.


# In[23]:


#Eigenvector Centrality

def eigen_central(G):
    eigen_centrality = nx.eigenvector_centrality(G, weight = "weight")
    
    eigen_centrality_sort = sorted(eigen_centrality.items(), key = lambda x:x[1], reverse = True)
    
    eigen_centrality_sort = pd.DataFrame.from_dict(eigen_centrality_sort)
    
    eigen_centrality_sort.columns = (["Character", "EigenVector Centrality"])
    
    return eigen_centrality_sort


# In[24]:


eigen_central(G)


# In[ ]:


#Observation:

#Tyrion Lannister is the character with the highest Degree Centrality, followed by Jon Snow and Jaime Lannister.
#The higher the number of connections, the higher the Degree Centrality.


# In[25]:


#Betweenness Centrality

def betweenness_central(G):
    betweenness_centrality = nx.betweenness_centrality(G, weight = "weight")
    
    betweenness_centrality_sort = sorted(betweenness_centrality.items(), key = lambda x:x[1], reverse = True)
    
    betweenness_centrality_sort = pd.DataFrame.from_dict(betweenness_centrality_sort)
    
    betweenness_centrality_sort.columns = (["Character", "Betweenness Centrality"])
    
    return betweenness_centrality_sort


# In[26]:


betweenness_central(G)


# In[ ]:


#However, when we look at Betweenness Centrality, it is Jon Snow who's at the top.

#So, Jon Snow is the central character that seems to best connect different, disparate groupings of characters.

#Note: The results may be different if we look at the individual books.


# In[ ]:


#Visualizing Graph Networks using Plotly
#Plotly is a data analytics and visualization library, that offers interactive visuals similar to Tableau
#& PowerBI. It is widely used in the Data Science community due to its interactivity and visual appeal.


# In[27]:


def draw_plotly_network_graph(Graph_obj, filter = None, filter_nodesbydegree = None):
    G_dup = Graph_obj.copy()

    degrees = nx.classes.degree(G_dup)
    
    degree_df = pd.DataFrame(degrees)
    
    if filter is not None:
        top = deg_centrality_sort[:filter_nodesbydegree]["Character"].values
        
        G_dup.remove_nodes_from([node
                             for node in G_dup.nodes
                             if node not in top
                            ]) # Filter out the nodes that fewer connections

    pos = nx.spring_layout(G_dup)

    for n, p in pos.items():
        G_dup.nodes[n]['pos'] = p

    edge_trace = go.Scatter(
        x = [],
        y = [],
        line = dict(width = 0.5, color = '#888'),
        hoverinfo = 'none',
        mode = 'lines')

    for edge in G_dup.edges():
        x0, y0 = G_dup.nodes[edge[0]]['pos']
        
        x1, y1 = G_dup.nodes[edge[1]]['pos']
        
        edge_trace['x'] += tuple([x0, x1, None])
        
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x = [],
        y = [],
        text = [],
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            showscale = True,
            colorscale = 'RdBu',
            reversescale = True,
            color = [],
            size = 15,
            colorbar = dict(
                thickness = 10,
                title = 'Node Connections',
                xanchor = 'left',
                titleside = 'right'
            ),
            line = dict(width = 0)))

    for node in G_dup.nodes():
        x, y = G_dup.nodes[node]['pos']
        
        node_trace['x'] += tuple([x])
        
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G_dup.adjacency()):
        node_trace['marker']['color'] += tuple([int(degree_df[degree_df[0] == adjacencies[0]][1].values)])
        
        node_info = adjacencies[0] + '<br /># of connections: ' + str(int(degree_df[degree_df[0] == adjacencies[0]][1].values))
        
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data = [edge_trace, node_trace],
                 layout = go.Layout(
                    title = '<br>GOT network connections',
                    titlefont = dict(size = 20),
                    showlegend = False,
                    hovermode = 'closest',
                    margin = dict(b = 20, l = 5, r = 5, t = 0),
                    annotations=[ dict(
                        text = "",
                        showarrow = False,
                        xref = "paper", yref = "paper") ],
                    xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
                    yaxis = dict(showgrid = False, zeroline = False, showticklabels = False)))

    iplot(fig)


# In[28]:


draw_plotly_network_graph(Graph_obj = G, filter = None, filter_nodesbydegree = None)

# Note: This cell will take sometime to run


# In[29]:


draw_plotly_network_graph(Graph_obj = G, filter = "Yes", filter_nodesbydegree = 50)


# In[30]:


draw_plotly_network_graph(Graph_obj = G1, filter = "Yes", filter_nodesbydegree = 50)


# In[31]:


deg_central(G1)[:20]


# In[33]:


draw_plotly_network_graph(Graph_obj = G2, filter = "Yes", filter_nodesbydegree = 50)


# In[34]:


deg_central(G2)[:20]


# In[35]:


draw_plotly_network_graph(Graph_obj = G3, filter = "Yes", filter_nodesbydegree = 50)


# In[36]:


deg_central(G3)[:20]


# In[37]:


draw_plotly_network_graph(Graph_obj = G4, filter = "Yes", filter_nodesbydegree = 50)


# In[38]:


deg_central(G4)[:20]


# In[39]:


betweenness_central(G4)[:20]


# In[40]:


draw_plotly_network_graph(Graph_obj = G5, filter = "Yes", filter_nodesbydegree = 50)


# In[41]:


deg_central(G5)[:20]


# In[42]:


betweenness_central(G5)[:20]


# In[43]:


# Creating a list of degree centrality of all the books
Books_Graph = [G1, G2, G3, G4, G5]

evol = [nx.degree_centrality(Graph) for Graph in Books_Graph]

# Creating a DataFrame from the list of degree centralities in all the books
degree_evol_df = pd.DataFrame.from_records(evol)

degree_evol_df.index = degree_evol_df.index + 1

# Plotting the degree centrality evolution of few important characters
fig = px.line(degree_evol_df[['Eddard-Stark', 'Tyrion-Lannister', 'Jon-Snow', 'Jaime-Lannister', 'Cersei-Lannister', 'Sansa-Stark', 'Arya-Stark']],
             title = "Evolution of Different Characters", width = 900, height = 600)

fig.update_layout(xaxis_title = 'Book Number',
                   yaxis_title = 'Degree Centrality Score',
                 legend = {'title_text': ''})

fig.show()


# In[45]:


import community as community_louvain
import matplotlib.cm as cm
import colorlover as cl


# In[46]:



# compute the best partition
partition = community_louvain.best_partition(G, random_state = 12345)

partition_df = pd.DataFrame([partition]).T.reset_index()

partition_df.columns = ["Character", "Community"]

partition_df


# In[47]:


partition_df["Community"].value_counts().sort_values(ascending = False)


# In[48]:


colors = cl.scales['12']['qual']['Paired']

def scatter_nodes(G, pos, labels = None, color = 'rgb(152, 0, 0)', size = 8, opacity = 1):
    # pos is the dictionary of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None, the Plotly's default color is used
    # size is the size of the dots representing the nodes
    # opacity is a value between 0 and 1, defining the node color opacity

    trace = go.Scatter(x = [], 
                    y = [],  
                    text = [],   
                    mode = 'markers', 
                    hoverinfo = 'text',
                           marker = dict(
            showscale = False,
            colorscale = 'RdBu',
            reversescale = True,
            color = [],
            size = 15,
            colorbar = dict(
                thickness = 10,
                xanchor = 'left',
                titleside = 'right'
            ),
            line = dict(width = 0)))
    
    for nd in G.nodes():
        x, y = G.nodes[nd]['pos']
        
        trace['x'] += tuple([x])
        
        trace['y'] += tuple([y])
        
        color = colors[partition[nd] % len(colors)]
        
        trace['marker']['color'] += tuple([color])
        
    for node, adjacencies in enumerate(G.adjacency()):
        node_info = adjacencies[0]
        
        trace['text'] += tuple([node_info])

    return trace    

def scatter_edges(G, pos, line_color = '#a3a3c2', line_width = 1, opacity = .2):
    trace = go.Scatter(x = [], 
                    y = [], 
                    mode = 'lines'
                   )
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        
        x1, y1 = G.nodes[edge[1]]['pos']
        
        trace['x'] += tuple([x0, x1, None])
        
        trace['y'] += tuple([y0, y1, None])
        
        trace['hoverinfo'] = 'none'
        
        trace['line']['width'] = line_width
        
        if line_color is not None:                 # when line_color is None, a default Plotly color is used
            trace['line']['color'] = line_color
    
    return trace


# In[49]:


def visualize_community(Graph, filter = "Yes", filter_nodes = 100):
    G_dup = G.copy()

    degrees = nx.classes.degree(G_dup)
    
    degree_df = pd.DataFrame(degrees)
    
    if filter is not None:
        top = deg_centrality_sort[:filter_nodes]["Character"].values
        
        G_dup.remove_nodes_from([node
                             for node in G_dup.nodes
                             if node not in top
                            ])

    pos = nx.spring_layout(G_dup, seed = 1234567)

    for n, p in pos.items():
        G_dup.nodes[n]['pos'] = p

    trace1 = scatter_edges(G_dup, pos, line_width = 0.25)
    trace2 = scatter_nodes(G_dup, pos)
    
    fig = go.Figure(data = [trace1, trace2],
             layout = go.Layout(
                title = '<br> GOT Community Detection',
                titlefont = dict(size = 20),
                showlegend = False,
                hovermode = 'closest',
                margin = dict(b = 20, l = 5, r = 5, t = 40),
                annotations = [ dict(
                    text = "",
                    showarrow = False,
                    xref = "paper", yref = "paper") ],
                xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
                yaxis = dict(showgrid = False, zeroline = False, showticklabels = False)))
     
    iplot(fig)


# In[50]:


visualize_community(Graph = G, filter = "Yes", filter_nodes = 100)


# In[ ]:




