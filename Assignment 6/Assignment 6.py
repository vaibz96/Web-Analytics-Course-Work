#!/usr/bin/env python
# coding: utf-8

# In[197]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.cluster import KMeansClusterer,cosine_distance
from sklearn.decomposition import LatentDirichletAllocation
import json
from numpy.random import shuffle


# In[138]:


def cluster_kmean(train_file,test_file):
    with open(train_file) as json_train_file:
        train_json_data = json.load(json_train_file)
        train_json_dataframe = pd.DataFrame(train_json_data)
        train_json_dataframe.columns = ['Text']
        #print(train_json_dataframe)

    with open(test_file) as json_test_file:
        test_json_data = json.load(json_test_file)
        test_json_dataframe = pd.DataFrame(test_json_data)
        test_json_dataframe.columns = ['Text','Labels']   
        test_json_dataframe['First'] = [x[0] for x in test_json_dataframe.Labels]
        unique_variety = test_json_dataframe["First"].unique()
#         print(unique_variety)
#         print(test_json_dataframe)
        
    # set the min document frequency to 5
    # generate tfidf matrix
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=5)

    dtm= tfidf_vect.fit_transform(train_json_dataframe['Text'])
    #print (dtm.shape)
    
    # set number of clusters
    num_clusters=3

    # initialize clustering model
    # using cosine distance
    # clustering will repeat 20 times
    # each with different initial centroids
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=20)

    # samples are assigned to cluster labels 
    # starting from 0
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)

    #print the cluster labels of the first 5 samples
    #print(clusters[0:5])
    
    # note transform function is used
    # not fit_transform
    test_dtm = tfidf_vect.transform(test_json_dataframe["Text"])

    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]

    #print(predicted[0:10])
    
    # determine cluster labels and calcuate precision and recall

    # Create a dataframe with cluster id and 
    # ground truth label
    confusion_df = pd.DataFrame(list(zip(test_json_dataframe['First'].values, predicted)),                            columns = ["label", "cluster"])
    confusion_df.head()

    # generate crosstab between clusters and true labels
    print(pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label))
    
    # Map cluster id to true labels by "majority vote"
    cluster_dict={i: j for i, j in enumerate(unique_variety)}
    print(cluster_dict)

    # Map true label to cluster id
    predicted_target=[cluster_dict[i] for i in predicted]

    print(metrics.classification_report(test_json_dataframe['First'], predicted_target))


# In[429]:


def cluster_lda(train_file,test_file):
    topic_assig=None
    labels=None
    
    train_data=json.load(open(train_file))
    test_data=json.load(open(test_file))
    
    train_data_text = list(train_data)
    test_data_text, test_data_label = zip(*test_data)
    first = [x[0] for x in test_data_label]

    tf_vectorizer = CountVectorizer(max_df=0.90, min_df=51, stop_words='english')
    train_tf = tf_vectorizer.fit_transform(train_data_text)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    num_topics = 3
    lda = LatentDirichletAllocation(n_components= num_topics, max_iter=20,                                     verbose=1, evaluate_every = 1, n_jobs=1, random_state=0).fit(train_tf)

#     num_top_words = 20
    
#     for topic_idx, topic in enumerate(lda.components_):
#         print ("Topic %d:" % (topic_idx))
#         # print out top 20 words per topic 
#         words=[(tf_feature_names[i],topic[i])for i in topic.argsort()[::-1][0:num_top_words]]
#         print(words)
#         print("\n")
      
    test_tf = tf_vectorizer.transform(test_data_text)
    topic_assign=lda.transform(test_tf)
    
    # set a probability threshold
        # the threshold determines precision/recall
    prob_threshold=0.25

    topics = np.copy(topic_assign)
    topics = np.where(topics >= prob_threshold, 1, 0)
    predicted = np.argmax(topics, axis=1).tolist()

    # Create a dataframe with cluster id and 
    # ground truth label
    confusion_df = pd.DataFrame(list(zip(first, predicted)), columns = ["label", "cluster"])
    #confusion_df.head()
    
    # Map cluster id to true labels by "majority
    confusion_df[['label']] = confusion_df[['label']].astype(str)
    crosstab = pd.crosstab( index = confusion_df.cluster, columns=confusion_df.label)
    print(crosstab)
    
    cluster_dict = crosstab.idxmax(axis=1).to_dict()
    # Map true label to cluster id
    predicted_target=[cluster_dict[i] for i in predicted]

    print(metrics.classification_report(first, predicted_target))
    return topic_assign, labels


# In[33]:


# def overlapping_cluster(topic_assign,labels):
#     final_thresh,f1 = None,None
#     # add your code here
#     return final_thresh,f1


# In[430]:


if __name__=="__main__":
    # Due to randomness, you won't get the exact result
    # as shown here, but your result should be close
    # if you tune the parameters carefully
    # Q1
    #cluster_kmean('C:\\Users\\shanb\PycharmProjects\\Web Mining\\Assignment 6\\train_text.json', 'C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 6\\test_text.json')
#     # Q2
     topic_assign, labels=cluster_lda('C:\\Users\\shanb\PycharmProjects\\Web Mining\\Assignment 6\\train_text.json', 'C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 6\\test_text.json')
#     # Q3
#     threshold,f1=overlapping_cluster(topic_assign,labels)
#     print(threshold)
#     print(f1)


# In[ ]:




