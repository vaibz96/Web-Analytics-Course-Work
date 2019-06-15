#!/usr/bin/env python
# coding: utf-8

# In[51]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import re    # import re module
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from scipy.spatial import distance
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


# In[52]:


def extract(text):
    text = re.sub(r"\n", "", text1)
    money = re.findall(r"\d{3}.\d{3}", text)
    year = re.findall(r'\d{4}', text)
    year_1 = [text.replace(')', '') for text in year]
    names = re.findall('[A-Za-z]+\s[A-Za-z]+\,', text)
    names_1 = [text.replace(',', '') for text in names]
    college = re.findall('\w+.\S\w+.\S\w+.\S+.?\s\(', text)
    college_1 = [text.strip("('").rstrip(',') for text in college]
    return list(zip(names_1, college_1, year_1, money))


# In[58]:


def word_pos_tag(pos_tag):
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'R'
    elif pos_tag.startswith('R'):
        # return wordnet tag "ADVERB"
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN


# In[72]:


def tokenize(doc, lemmatized=False, no_stopword=False):
    tokens = []
    
    tokens = [word for word in nltk.word_tokenize(doc)]

    
    stop_words = stopwords.words('english')
    
    
    if no_stopword == True:
        tokens = [word for word in nltk.word_tokenize(doc)                  if word not in stop_words] 
        
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tag = nltk.pos_tag(tokens)
    
    if lemmatized == True:
        tokens = [wordnet_lemmatizer.lemmatize(word, word_pos_tag(tag))                   for (word, tag) in word_tag]    
    
    return tokens


# In[73]:


def get_similarity(q1, q2, lemmatized=False, no_stopword=False):
    sim = []
    
    for i in range(0, len(q1)):
        q1_tokenize = tokenize(q1[i], lemmatized, no_stopword)
        q2_tokenize = tokenize(q2[i], lemmatized, no_stopword)
        
        token_count_q1=nltk.FreqDist(q1_tokenize)
        token_count_q2=nltk.FreqDist(q2_tokenize)
        
        docs_tokens={0:token_count_q1, 1:token_count_q2}
        dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
        dtm=dtm.fillna(0)  
        # get normalized term frequency (tf) matrix        
        tf=dtm.values
        doc_len=tf.sum(axis=1)
        tf=np.divide(tf.T, doc_len).T
        # get idf
        df=np.where(tf>0,1,0)
        #idf=np.log(np.divide(len(docs), \
        #    np.sum(df, axis=0)))+1

        smoothed_idf=np.log(np.divide(len(q1+q2)+1, np.sum(df, axis=0)+1))+1    
        smoothed_tf_idf=tf*smoothed_idf
        
        similarity=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))  
        sim.append((similarity[0][1]))
    
    #print(sim)
    return sim


# In[74]:


def predict(sim, ground_truth, threshold=0.5):
    predict=[]
    num=[]
    
    is_duplicate_count = ground_truth.sum(axis=0)
    
    for i in sim:
        if i > threshold:
            predict.append(1) 
        else:
            predict.append(0)
            
    for i in range(0, len(predict)):    
        if predict[i] == 1 and ground_truth[i] == 1:
            num.append(1)
        
    recall = ((np.sum(num))/is_duplicate_count)
    return predict,recall


# In[75]:


if __name__ == '__main__':
    text1 = '''Following is total compensation for other presidents at private colleges in Ohio in 2015:
    Grant Cornwell, College of Wooster (left in 2015): $911,651
    Marvin Krislov, Oberlin College (left in 2016):  $829,913
    Mark Roosevelt, Antioch College, (left in 2015): $507,672
    Laurie Joyner, Wittenberg University (left in 2015): $463,504
    Richard Giese, University of Mount Union (left in 2015): $453,800'''

    print(extract(text1))
    
    data = pd.read_csv("C:/Users/shanb/PycharmProjects/Web Mining/quora_duplicate_question_500.csv",header=0)
    #data.head(3)
    q1 = data['q1'].values.tolist()
    q2 = data['q2'].values.tolist()
    print("Test Q1")
    print("\nlemmatized: No, no_stopword: No")
    sim=get_similarity(q1,q2)
    #print(sim)
    pred, recall=predict(sim,data["is_duplicate"].values)
    print(recall)
    
    print("\nlemmatized: Yes, no_stopword: No")
    sim=get_similarity(q1,q2,True)
    #print(sim)
    pred, recall=predict(sim,data["is_duplicate"].values)
    print(recall)
    
    print("\nlemmatized: No, no_stopword: Yes")
    sim=get_similarity(q1,q2,False,True)
    #print(sim)
    pred, recall=predict(sim,data["is_duplicate"].values)
    print(recall)
    
    print("\nlemmatized: Yes, no_stopword: Yes")
    sim=get_similarity(q1,q2,True,True)
    #print(sim)
    pred, recall=predict(sim,data["is_duplicate"].values)
    print(recall)


# In[ ]:





# In[ ]:




