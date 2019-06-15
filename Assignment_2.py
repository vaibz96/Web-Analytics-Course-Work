# Structure of your solution to Assignment 1
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import numpy as np
import time


def analyze_data(filepath):
    filepath = r'question.csv'

    df = pd.read_csv(filepath)
    print(df[(df.answercount > 0)].sort_values(by=['viewcount'], ascending=[True]).tail(3)[['title', 'viewcount']],
          '\n')
    print(df.groupby('quest_name').size().sort_values().tail(5), '\n')
    df['first_tag'] = df[['tags']].apply(lambda col: df['tags'].str.split(',').str[0], axis=0)
    print(df, '\n')
    grouped = df.groupby(['first_tag'])
    print(grouped['viewcount'].agg([np.mean, np.max, np.min]).loc[['python', 'pandas', 'dataframe'], :], '\n')
    print(pd.crosstab(index=[df.answercount], columns=[df.first_tag], values=df.answercount,aggfunc='count', margins=True), '\n')


def analyze_tf_idf(arr, K):
    tf_ifd = None
    top_k = None
    ld = np.sum(arr, axis=1)
    tf = []
    tf.append(arr[0] / ld[0])
    tf.append(arr[1] / ld[1])
    tf.append(arr[2] / ld[2])
    df = np.sum((arr > 0), axis=0)
    tf_idf = tf / (np.log(df) + 1)
    tf_idf
    top_k = (-tf_idf).argsort()[:K, :3]  # [:,:3]
    start = time.time()  # get starting time
    for row in tf:
        row += df
        print("time used for rows to generate: %.4f ms" % (time.time() - start))
    return tf_idf, top_k


# def analyze_corpus(filepath):

# add your code here


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":
    # Test Question 1
    arr = np.array([[0, 1, 0, 2, 0, 1], [1, 0, 1, 1, 2, 0], [0, 0, 2, 0, 0, 1]])

    print("\nQ1")
    tf_idf, top_k = analyze_tf_idf(arr, 3)
    print(tf_idf)
    print(top_k)

    print("\nQ2")
    print(analyze_data('question.csv'))

    # test question 3
#    print("\nQ3")
#    analyze_corpus('question.csv')