#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Context
#The discovery of DNA (Deoxyribonucleic Acid), and the critical role it plays in information 
#storage for all biological beings, was a seminal moment for the biological sciences. All 
#the information that is needed for the functioning of a living cell is encoded in and ultimately 
#derived from the DNA of that cell, and this holds true for all biological organisms on the planet.

#DNA can be represented as a text sequence, with an alphabet that only has four letters - A (Adenosine), 
#C (Cytosine), G (Guanine), and T (Thymine). The diversity of living organisms and their complex properties 
#is hidden in their genomic sequences. One of the most exciting problems in modern science is to understand
#the organization of living matter by reading genomic sequences.

#One distinctive message in a genomic sequence is a piece of text, called a gene. Genes can be oriented
#in the sequence in either the forward or backward directions. In the highest organisms (humans, for example),
#the notion of a gene is more complex.

#It was one of the many great discoveries of the Twentieth century, that biological information is
#encoded in genes through triplets of letters, called codons in the biological literature.


# In[ ]:


#ABout the data
#The work starts with a fragment of the genomic sequence of the bacterium Caulobacter Crescentus. 
#This sequence is given as a long text file (300 kb), and the task is to look at the file and ensure 
#that the text uses the alphabet of four letters (A, C, G, and T) and that these letters are used 
#without spaces. It is noticeable that, although the text seems to be random, it is well organized, 
#but we cannot understand it without special tools. Statistical methods may help us do so.


# In[ ]:


#objective
#In this exercise, we will see that it is possible to verify the validity of the discovery of
#three-letter codons, simply by performing unsupervised machine learning on the genetic sequence.

#In this case study, we accept data from a genome and have the goal of identifying useful genes
#versus noise. Unfortunately, we don't know which sequences of genes are useful, so we have to 
#use Unsupervised Learning to infer this.

#In this notebook, we walk through the following series of steps:

#First, the data is imported and prepared. Initially, the sequence, a single string, is split into 
#non-overlapping substrings of length 300, and we may then count the combinations of the distinct 
#1, 2, 3, and 4-length sequences of base pairs that appear in each possible substring.
#PCA is performed to try to identify the internal structure of the data.
#Finally, if PCA reveals some internal structure then we'll apply Clustering techniques to the dataset.


# In[1]:


import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans 

from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[2]:


#Data Preparation
#The file format often used for bioinformatics and genomic data is called FASTA. It is a normally 
#encoded file with headers separating genetic information. We read the file and strip it of unwanted 
#characters and store it as a string.


# In[3]:


# Open the file and get an array of its lines 
with open ("/Users/yutaoyan/Desktop/Genomics/ccrescentus.fa", "r") as inputFile:
    data = inputFile.readlines()

# Concatenate each line from the second (first line is a description), stripped of empty characters 
geneticCode = ''

for line in data[1:]:
    geneticCode += line.strip()
    
# Count the presence of each genome (a, g, t, c)
aCount = geneticCode.count('a')

gCount = geneticCode.count('g')

tCount = geneticCode.count('t')

cCount = geneticCode.count('c')

# For testing, we print a sample of the string and check if there are no wanted characters
print(geneticCode[0:30])

print('Test: only a, g, t, c?')

print(aCount + gCount + tCount + cCount == len(geneticCode))


# In[4]:


# Size of the sub strings (data points)
size = 300

dataPoints = []

# Copy the entire code into a string, which will be removed of its first elements iteratively
tempString = geneticCode

# Iteratively remove a left chunk of the string and place it into our array
while len(tempString) > 0:
    dataPoints.append(tempString[0: size])
    
    tempString = tempString[size:]

print(dataPoints[0])


# In[5]:


import itertools

iterables = ['a', 'g', 't', 'c']

wordsDict =  {}

# For words of size 1 to 4, we calculate the cartesian product to get all the possibilities
for i in range(1, 5):
    words = []
    
    iterator = itertools.product(iterables, repeat = i)
    
    for word in iterator:
        s = ''
        for t in word:
            s += t
        words.append(s)
    wordsDict[i] = words

# Print the dictionary for 3 letter words
print(wordsDict[3])


# In[6]:


# Dictionary that will contain the frequency table for each word size
freqTables = {}

for i in range(1, 5):
    df = pd.DataFrame(columns = wordsDict[i])       # Create an empty DataFrame with columns being the words on the dictionary
    
    for index, dataP in enumerate(dataPoints):
        
        df.loc[index] = np.zeros(len(wordsDict[i])) # Create a row with zero values corresponding to a data point
        
        while len(dataP) > 0:
        
            left = dataP[0:i]                       # Get the left part of the data point (i characters)
            
            df.loc[index, left] += 1                # Find it in the respective column and count it there
            
            dataP = dataP[i:]
    
    freqTables[i] = df

freqTables[3].head()


# In[8]:


normFreqTables = {}

for i in range(1, 5):
    # Eliminate the string column from the data, leaving only the actual frequencies
    data = freqTables[i]

    data = StandardScaler(copy = True, with_mean = True, with_std = True).fit_transform(data)
    
    normFreqTables[i] = pd.DataFrame(data, columns = wordsDict[i])

# For testing, we check that the average of a column is close to zero and the stdev is close to 1
print(normFreqTables[2].loc[:, 'gt'].mean())

print(normFreqTables[2].loc[:, 'gt'].std())


# In[9]:


pca = PCA(n_components = 2)

pCompTables = {}

for i in range(1, 5):
    pca.fit(normFreqTables[i])
    
    pComponents = pca.transform(normFreqTables[i])
    
    # For each word size, we store the result of the PCA in a table containing only the first two princicipal components
    pCompTables[i] = pd.DataFrame(pComponents[:, [0, 1]], columns = ['pc1', 'pc2'])
    
    print('Explained variance for ' + str(i) + ' letters: ' + str(pca.explained_variance_ratio_.sum()))
    
print(pCompTables[2].head())


# In[10]:


# Now, we finally need to plot these tables to try to find correlations visually
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for i in range(1, 5):
    plt.subplot(2, 2, i)
    
    x = pCompTables[i].loc[:,'pc1']
    
    y = pCompTables[i].loc[:,'pc2']
    
    plt.scatter(x, y, s = 1)
    
    plt.xlabel('pc1')
    
    plt.ylabel('pc2')
    
    plt.title(str(i) + ' letter words')

plt.show()


# In[ ]:


#Clustering
#We will now cluster the 3 letter word gene breakdown using the K-means Clustering unsupervised 
#algorithm. From the previous section, we can detect 6 or 7 clusters. Knowing that some genes do
#not carry information, we are led to think that the center points, far from the 6 distinct centroids,
#could be a candidate for those. We, therefore, assume that there are 7 clusters (this could be 
#checked by comparing the performance with 6 or 7 clusters).


# In[11]:


kmeans = KMeans(n_clusters = 7)

kmeans.fit(normFreqTables[3])


# In[ ]:


#Results in a Graph
#Using our clustering results, we can visualize the different colors!


# In[12]:


plt.figure(figsize = (8, 8))

x = pCompTables[3].loc[:,'pc1']

y = pCompTables[3].loc[:,'pc2']

plt.scatter(x, y, s = 20, c = kmeans.labels_, cmap = 'rainbow')

plt.xlabel('pc1')

plt.ylabel('pc2')

plt.title('K-Means clustering showing on top of principal components')

plt.show()


# In[ ]:


#Conclusion
#Hence, Unsupervised Learning through Clustering (K-means Clustering) and Dimensionality 
#Reduction (PCA) have allowed us to visualize, validate, and provide supporting evidence for 
#the biological discovery that the DNA genetic sequence is organized into three-letter words 
#called codons, which are responsible for the amino acids and proteins that are produced by living cells.

