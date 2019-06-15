#!/usr/bin/env python
# coding: utf-8

# In[305]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import nltk
import numpy as np
# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# import pipeline class
from sklearn.pipeline import Pipeline
# import GridSearch
from sklearn.model_selection import GridSearchCV
# import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
# import method for split train/test data set
from sklearn.model_selection import train_test_split
# import method to calculate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# In[350]:


def classify(train_file, test_file):
    train_data = pd.read_csv(train_file, header=0)
    test_data = pd.read_csv(test_file, header=0)

    # Exercise 3.3.1 Grid search    
    
    # build a pipeline which does two steps all together:
    # (1) generate tfidf, and (2) train classifier
    # each step is named, i.e. "tfidf", "clf"

    text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', MultinomialNB())])

    # set the range of parameters to be tuned
    # each parameter is defined as 
    # <step name>__<parameter name in step>
    # e.g. min_df is a parameter of TfidfVectorizer()
    # "tfidf" is the name for TfidfVectorizer()
    # therefore, 'tfidf__min_df' is the parameter in grid search

    parameters = {'tfidf__min_df':[1, 2, 3],
                  'tfidf__stop_words':[None,"english"],
                  'clf__alpha': [0.5,1.0,2.0],
    }

    # the metric used to select the best parameters
    metric =  "f1_macro"

    # GridSearch also uses cross validation
    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)
    #print(gs_clf)

    gs_clf_best = gs_clf.fit(train_data["text"], train_data["label"])
    #print(gs_clf_best)
    for param_name in gs_clf_best.best_params_:
        print(param_name,": ",gs_clf_best.best_params_[param_name])

    print("best f1 score:", gs_clf_best.best_score_)
    
    # take the parameters to calculate the tf-idf
    tfidf__min_df = gs_clf.best_params_['tfidf__min_df']
    tfidf__stop_words = gs_clf.best_params_['tfidf__stop_words']
    clf__alpha = gs_clf.best_params_['clf__alpha']
    
    # initialize the TfidfVectorizer 
    tfidf_vect = TfidfVectorizer(stop_words= tfidf__stop_words, min_df= tfidf__min_df)
    
    # generate tfidf matrix
    dtm = tfidf_vect.fit_transform(train_data["text"])
    dtm_test = tfidf_vect.transform(test_data["text"])
    
    X_train = dtm
    y_train = train_data['label']
    
    X_test = dtm_test
    y_test = test_data['label']
    
    # train a multinomial naive Bayes model using the testing data
    clf = MultinomialNB(alpha = clf__alpha).fit(X_train, y_train)

    # predict the news group for the test dataset
    predicted = clf.predict(X_test)
    
    # get the list of unique labels
    labels=sorted(test_data["label"].unique())
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, predicted, labels=labels)
#     print("labels: ", labels)
#     print("precision: ", precision)
#     print("recall: ", recall)
#     print("f-score: ", fscore)
#     print("support: ", support)
    print(classification_report(y_test, predicted))
    
    #AUC
    # We need to get probabilities as predictions
    predict_p=clf.predict_proba(X_test)
    # a probability is generated for each label
    #print(labels)
    #print(predict_p[0:3])
    # Ground-truth
    # let's just look at one label "2"
    # convert to binary
    binary_y = np.where(y_test==2,1,0)
    # this label corresponds to last column
    y_pred = predict_p[:,1]
    # compute fpr/tpr by different thresholds
    # positive class has label "1"
    
    fpr, tpr, thresholds = roc_curve(binary_y, y_pred, pos_label=1)
    
    # calculate auc
    auc(fpr, tpr)
    
    plt.figure();
    plt.plot(fpr, tpr, color='darkorange', lw=3);
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('AUC of Naive Bayes Model');
    plt.show();
    
    # compute precision/recall by different thresholds
    precision, recall, thresholds = precision_recall_curve(binary_y, y_pred, pos_label=1)

    plt.figure();
    plt.plot(recall, precision, color='darkorange', lw=3);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title('Precision_Recall_Curve of Naive Bayes Model');
    plt.show();
    
    
def impact_of_sample_size(train_file):
    train_large_data = pd.read_csv(train_file,header=0)
#     print(train_large_data.head())
#     print(len(train_large_data))
    sample_size = []
    F1_NB = []
    F1_SVM =[]
    AUC_NB = []
    AUC_SVM = []
    for i in range(0, len(train_large_data)-799, 400):
        train_data = train_large_data[:800+i]
        #print("length",len(train_data))
        sample_size.append(len(train_data))
    #print(sample_size)
        # initialize the TfidfVectorizer 
        tfidf_vect = TfidfVectorizer(stop_words= "english")
        # generate tfidf matrix
        dtm= tfidf_vect.fit_transform(train_data["text"])
        metrics = ['precision_macro', 'recall_macro', "f1_macro", "roc_auc"]
        clf = MultinomialNB()
        binary_y= np.where(train_data["label"]==2,1,0)
        cv = cross_validate(clf, dtm, binary_y, scoring=metrics, cv=5, return_train_score=True)
        #store the result in an list
        F1_NB.append(cv['test_f1_macro'].mean())
        AUC_NB.append(cv['test_roc_auc'].mean())
        
        # initiate a linear SVM model
        clf_svm = svm.LinearSVC()
        cv_svm = cross_validate(clf_svm, dtm, binary_y,scoring = metrics, cv = 5)
        #store the result in list
        F1_SVM.append((cv_svm['test_f1_macro'].mean()))
        AUC_SVM.append((cv_svm['test_roc_auc'].mean()))
    
    plt.figure();
    plt.plot(sample_size, F1_NB);
    plt.plot(sample_size, F1_SVM);
    plt.xlabel('Size');   
    plt.legend(['F1_NB','F1_SVM'],loc=0);
    plt.show();
    
    plt.figure();
    plt.plot(sample_size, AUC_NB);
    plt.plot(sample_size, AUC_SVM);
    plt.xlabel('Size');
    plt.legend(['AUC_NB','AUC_SVM'],loc=0);
    plt.show();
       
def classify_duplicate(filename):
    filename_data = pd.read_csv(filename,header=0)
    data = []

    for i in range(0, len(filename_data)):
        data.append(filename_data.iloc[i,0]+' '+filename_data.iloc[i,1])
    #print(data) 
    # initialize the TfidfVectorizer
    tfidf_vect = TfidfVectorizer(stop_words = "english", smooth_idf =True).fit(data)
    # generate tfidf matrix for both the questions
    dtm_q1 = tfidf_vect.transform(filename_data["q1"])
    dtm_q2 = tfidf_vect.transform(filename_data["q2"])
    data_1 = []
    for i in range(0, len(filename_data)):
        cosine_sim = cosine_similarity(dtm_q1[i], dtm_q2[i])[0]
        tokenized_corpus = [doc.split(" ") for doc in filename_data["q1"]]
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(filename_data.iloc[i,1].split(" "))[i] 
        data_1.append([cosine_sim, doc_scores])
    #print(data_1)
    metrics = ["roc_auc"]
    binary_y= np.where(filename_data["is_duplicate"]==0,1,0)
    # initiate a linear SVM model
    clf_svm = svm.LinearSVC()
    cv_svm = cross_validate(clf_svm, data_1, binary_y, scoring = metrics, cv = 5)
    
    auc = cv_svm['test_roc_auc'].mean()
    return auc


# In[351]:


if __name__ == "__main__":
    # Question 1
    # Test Q1
    
    classify("C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 5\\train.csv",             "C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 5\\test.csv")
    # Test Q2
    
    impact_of_sample_size("C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 5\\train_large.csv")
    
    # Test Q3
    
    result = classify_duplicate("C:\\Users\\shanb\\PycharmProjects\\Web Mining\\Assignment 5\\quora_duplicate_question_500.csv")
    print("Q3: ", result)


# In[ ]:





# In[ ]:




