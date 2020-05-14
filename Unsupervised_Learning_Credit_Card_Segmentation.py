#!/usr/bin/env python
# coding: utf-8

# ### ALLLIFE BANK CREDIT CARD CUSTOMER SEGMENTATION
# 
# DATA: 
# 
# * VARIOUS CUSTOMERS OF BANK
# * CUSTOMERS' CREDIT LIMIT
# * CUSTOMERS' TOTAL NUMBER CREDIT CARDS
# * DIFFERENT CONTACT CHANNELS FOR QUERIES (BANK VISITS, ONLINE, CALL CENTRE)
# 
# 
# KEY FOCUS
# 
# * DIFFERENT SEGMENTS OF CUSTOMERS
# * DIFFERENCES BETWEEN SEGMENTS 

# In[1]:


### LIBRARIES

import pandas as pd

from sklearn.preprocessing import StandardScaler

import seaborn as sns 
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np 

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,cophenet


# In[2]:


### READ DATASET
df=pd.read_excel('Credit Card Customer Data.xlsx')


# In[4]:


### VIEW 
### (2) - TOP TWO ROWS

df.head(2)


# In[5]:


# SI_NO & CUSTOMER KEY ARE UNIQUE VARIABLES
# REMOVE SI_NO & CUSTOMER KEY AS THEY WILL NOT PLAY ANY NECESSARY ROLE IN CLUSTERING


# In[6]:


cols_to_consider=['Avg_Credit_Limit','Total_Credit_Cards','Total_visits_bank','Total_visits_online','Total_calls_made']


# In[7]:


subset=df[cols_to_consider]


# In[8]:


#### EDA ####


# In[9]:


### MISSING VALUE CHECK
subset.isna().sum() 


# In[10]:


# NO MISSING VALUES IN DATA


# In[11]:


subset.describe()


# In[12]:


# AVG_CREDIT_LIMIT MIN & MAX SUBSTANTIALLY LARGER THAN OTHER VARIABLES
# DATA MUST BE STANDARDIZED IN ORDER TO BE EQUAL IN SCALE


# In[13]:


### Z-SCORE -> STANDARD SCALER

scaler=StandardScaler()
subset_scaled=scaler.fit_transform(subset)   


# In[17]:


subset_scaled_df=pd.DataFrame(subset_scaled,columns=subset.columns)
subset_scaled_df.describe()


# In[18]:


#### VIZUALISATION OF DATA ####


# In[19]:


### CORRELATION HEATMAP
sns.heatmap(subset_scaled_df.corr())


# In[20]:


# NO SIGNIFICANT CORRELATIONS


# In[21]:


### PAIRPLOT
sns.pairplot(subset_scaled_df,diag_kind="kde")


# In[22]:


# DATA SEEMS TO BE A MIXTURE OF GAUSSIANS


# In[23]:


### ELBOW PLOT

clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction=model.predict(subset_scaled_df)
    distortion=sum(np.min(cdist(subset_scaled_df, model.cluster_centers_, 'euclidean'), axis=1)) / subset_scaled_df.shape[0]
                           
    meanDistortions.append(distortion)

    print(k,distortion)
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')


# In[24]:


# K-MEANS WITH K = 3
kmeans = KMeans(n_clusters=3, n_init = 15, random_state=2345)
kmeans.fit(subset_scaled_df)


# In[25]:


centroids = kmeans.cluster_centers_
centroids


# In[26]:


centroid_df = pd.DataFrame(centroids, columns = subset_scaled_df.columns )


# In[28]:


# CENTROIDS FOR DIFFERENT CLUSTERS
centroid_df


# In[29]:


### AD LABELS

### CREATE COPY OF DATA
dataset=subset_scaled_df[:]

dataset['KmeansLabel']=kmeans.labels_

dataset.head(10)


# In[30]:


#### CLUSTER VIZUALISATION ####


# In[31]:


### SCATTERPLOTS
### 5 DIMENSIONS IN DATASET
### CREATE SCATTERPLOTS FROM 2 RANDOM FEATURES

plt.scatter(dataset['Avg_Credit_Limit'], dataset['Total_Credit_Cards'], c=kmeans.labels_,)  
plt.show()


# In[32]:


plt.scatter(dataset['Avg_Credit_Limit'], dataset['Total_visits_bank'], c=kmeans.labels_,)  
plt.show()


# In[33]:


plt.scatter(dataset['Avg_Credit_Limit'], dataset['Total_calls_made'], c=kmeans.labels_,)  
plt.show()


# In[34]:


#### ANALYZE CLUSTERS ####


# In[35]:


### BOXPLOTS TO OBSERVE STATISTICAL PROPERTIES

dataset.boxplot(by = 'KmeansLabel',  layout=(2,4), figsize=(20, 15))
plt.show()


# In[36]:


# DIFFERENTIATED CLUSTERS OBSERVABLE


# In[37]:


#### HIERARCHICAL CLUSTERING ####


# In[38]:


### DENDOGRAMS
### COPHENETIC COEFFICIENT 

linkage_methods=['single','complete','average','ward','median']
results_cophenetic_coef=[]
for each in linkage_methods :
    Z = linkage(subset_scaled_df, method=each, metric='euclidean')
    cc,cophn_dist=cophenet(Z,pdist(subset_scaled_df))
   
    #Plotting the dendogram 
    plt.figure(figsize=(25, 10))
    dendrogram(Z)
    plt.title("Linkage Type: "+each +"having cophenetic coefficient : "+str(round(cc,3)) )
    plt.show()
    
    results_cophenetic_coef.append((each,cc))
    print (each,cc)


# In[39]:


results_cophenetic_coef_df=pd.DataFrame(results_cophenetic_coef,columns=['LinkageMethod','CopheneticCoefficient'])
results_cophenetic_coef_df


# In[40]:


# 'AVERAGE' LINKAGE YIELDS BEST RESULT


# In[41]:


### DENDOGRAM FOR LAST 20 FORMED CLUSTERS
### TRUNCATE_MODE: 'lastp'

Z = linkage(subset_scaled_df, 'average', metric='euclidean')

dendrogram(
    Z,
    truncate_mode='lastp',  
    p=20
)
plt.show()


# In[42]:


### MAX DISTANCE: 3.2 
### MAX DISTANCE 3.2 TO FORM DIFFERENT CLUSTERS

max_d=3.2
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, max_d, criterion='distance')


# In[43]:


set(clusters)


# In[44]:


# 3 CLUSTERS FORMED


# In[45]:


#### CLUSTERS LABEL TO DATA SET ####


# In[46]:


### COPY OF DATASET
dataset2=subset_scaled_df[:]

dataset2['HierarchicalClusteringLabel']=clusters

dataset2.head(3)


# In[47]:


#### ANALYZE CLUSTERS ####


# In[48]:


dataset2.boxplot(by = 'HierarchicalClusteringLabel',  layout=(2,4), figsize=(20, 15))
plt.show()


# In[49]:


#### K-MEANS VS HIERARCHICAL RESULTS ####


# In[50]:


Kmeans_results=dataset.groupby('KmeansLabel').mean()
Kmeans_results


# In[51]:


Hierarchical_results=dataset2.groupby('HierarchicalClusteringLabel').mean()
Hierarchical_results


# In[52]:


# SIMILARITIES:
    # * CLSTR 0 OF KM & CLSTR 2 OF H --> RENAME AS G1
    # * CLSTR 1 OF KM & CLSTR 3 OF H --> RENAME AS G2
    # * CLSTR 2 OF KM & CLSTR 1 OF H --> RENAME AS G3


# In[53]:


Kmeans_results.index=['G1','G2','G3']
Kmeans_results


# In[54]:


Hierarchical_results.index=['G3','G1','G2']
Hierarchical_results.sort_index(inplace=True)
Hierarchical_results


# In[55]:


Kmeans_results.plot.bar()


# In[56]:


Hierarchical_results.plot.bar()


# In[57]:


#### MARKETING RECOMMENDATION & CLUSTER PROFILES ####


# In[58]:


# BOTH CLUSTERING METHODS PROVIDED SIMILAR CLUSTERS
# LABELS CAN BE ASSIGNED FROM EITHER ALGORITHM TO UNSCALED ORIGINAL DATA


# In[59]:


subset['KmeansLabel']=dataset['KmeansLabel']


# In[60]:


for each in cols_to_consider:
    print (each)
    print ( subset.groupby('KmeansLabel').describe().round()[each][['count','mean','min','max']])
    
    print ("\n\n")

