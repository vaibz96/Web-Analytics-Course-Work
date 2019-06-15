#!/usr/bin/env python
# coding: utf-8

# In[367]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import requests
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import re
from fractions import Fraction


page = requests.get('https://www.rottentomatoes.com/m/finding_dory/reviews/')

# Q1
def getData(movie_id):
    data=[]
    if page.status_code==200:        
        soup = BeautifulSoup(page.content, 'html.parser')
    
        # find a block with id='seven-day-forecast-body'
        # follow the path down to the div for each period
        divs=soup.select("div.col-xs-16.review_container div.review_area")
        #print(len(divs))
        #print divs
    
        for idx, div in enumerate(divs):
        # for testing you can print idx, div
        #print idx, div 
        
        # initiate the variable for each period
            Date=None
            Review=None
            Ratings=None
        
        # get Date
            Date=div.select("div.review_date.subtle.small")[0].text.strip()
            
        # get Review
            Review=div.select("div.the_review")[0].text.strip()
        
        # get Ratings
            Ratings=div.select("div.review_desc div.small.subtle")[0].text.replace("Full Review | Original Score: ", "")            .rstrip("Full Review").strip()
            if Ratings == "":
                Ratings= None
        # add date, review, and rating as a tuple into the list
            data.append((Date, Review, Ratings))
            #print((Date, Review, Ratings))    
    return data

#Q2
def plot_data(data):
    ratings = []
    year = []
    data = getData("Fiding Dory")
    for i in data:
        ratings.append(i[2])
        year.append(i[0][-4:])
        
    movie_ratings=[]
    for x in ratings:
        if x == None:
            x="0/1"
        elif x == "B":
            
            x="0/1"
        elif x == "B+":
            
            x="0/1"
        num, denom = x.split("/")
        num, denom = float(num), float(denom)
        float_ratings = num/denom
        movie_ratings.append(float_ratings)
    data = { 'year': year, 'ratings': movie_ratings}
    df = pd.DataFrame(data)
    plt.bar(df.year, df.ratings, color='blue')
    plt.xlabel("Year")
    plt.ylabel("Ratings")
    plt.title("Average rating by year")
    plt.show()
     
             
# Q3   
def getFullData(movie_id):
    data1=[]
    page = requests.get('https://www.rottentomatoes.com/m/finding_dory/reviews/')
    soup = BeautifulSoup(page.content, 'html.parser')
    span = soup.select("span.pageInfo")[0].text
    page_number = re.findall('1[0-9]', span)
    pg_num = "".join(map(str, page_number))
    #print(pg_num)
    pgNums = int(pg_num)
    for i in range(1, pgNums+1):
        url = 'https://www.rottentomatoes.com/m/finding_dory/reviews/?page={}&sort='.format(i)
        page = requests.get(url)
        if page.status_code == 200:        
            soup = BeautifulSoup(page.content, 'html.parser')
    
    # find a block with class='div.col-xs-16.review_container div.review_area'
    # follow the path down to the div for each period
            divs = soup.select("div.col-xs-16.review_container div.review_area")
        #print(len(divs))
    #print divs
    
            for idx, div in enumerate(divs):
        # for testing you can print idx, div
        #print idx, div 
        
        # initiate the variable for each period
                Date = None
                Review = None
                Ratings = None
        
        # get Date
                Date = div.select("div.review_date.subtle.small")[0].text.strip()
                if Date == "":
                    Date = None
            
        # get Review
                Review = div.select("div.the_review")[0].text.strip()
                if  Review == "":
                    Review = None
        
        # get Ratings
                Ratings=div.select("div.review_desc div.small.subtle")[0].text.replace("Full Review | Original Score: ", "").rstrip("Full Review").strip()
                if Ratings == "":
                    Ratings = None
        # add date, review, and rating as a tuple into the list
                data.append((Date, Review, Ratings))
                #print((Date, Review, Ratings)) 
    return data

if __name__ == "__main__":
    # Test Q1
    data=getData("Finding_Dory")
    print(data, "\n")
    # Test Q2
    plot_data(data)
    # Test Q3
    data=getFullData("Finding_Dory")
    print(len(data), data[-1])
    #plot_data(data)


# In[ ]:





# In[ ]:




