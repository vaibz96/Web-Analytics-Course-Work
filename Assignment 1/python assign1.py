#!/usr/bin/env python
# coding: utf-8

# Structure of your solution to Assignment 1

import csv
import string
from collections import Counter
import re

import operator
def tokenize(text):
    tokens=[]
    text = text.lower()
    text = text.replace('\n', ' ')
    tokens = text.split()
    tokens = [i.strip(string.punctuation) for i in tokens]
    return tokens

class Text_Analyzer():
    def __init__(self, text):

        self.text=text
        token_count={}


    def analyze(self):
        tokens=[]
        tokens=tokenize(text)
        token_count = {x: tokens.count(x) for x in set(tokens)}
        return token_count

    def topN(self, N):
        self.N = int(N)
        tokens = tokenize(text)
        token_count = {x: tokens.count(x) for x in set(tokens)}
        sorted_data=sorted(token_count.items(), key=lambda x:x[0])

        print(sorted_data,"\n")
        print("Output for topN","\n")
        top_N= list(tuple(Counter(token_count).most_common(N)))
        return top_N

def bigram(text, N):
    
    result = []
    words = tokenize(text)

    bigrams= [[item[0],item[1]] for item in zip(words,words[1:])]
    print(bigrams,"\n")
    print(Counter(zip(words, words[1:])),"\n")
    result = list(Counter(zip(words, words[1:])).most_common(N))
    return result

    return result



if __name__ == "__main__":
    # Test Question 1
    text = ''' There was nothing so VERY remarkable in that; nor did Alice
think it so VERY much out of the way to hear the Rabbit say to
itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought
it over afterwards, it occurred to her that she ought to have
wondered at this, but at the time it all seemed quite natural);
but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-
POCKET, and looked at it, and then hurried on, Alice started to
her feet, for it flashed across her mind that she had never
before seen a rabbit with either a waistcoat-pocket, or a watch to
take out of it, and burning with curiosity, she ran across the
field after it, and fortunately was just in time to see it pop
down a large rabbit-hole under the hedge.
'''
    print(tokenize(text),"\n")

    # Test Question 2
    analyzer = Text_Analyzer(text)
    print(analyzer.analyze(),"\n")
    
    print(analyzer.topN(5),"\n")


    # 3 Test Question 3

    top_bigrams = bigram(text, 6)
    print("Output for bigrams\n")
    print(top_bigrams)



        

