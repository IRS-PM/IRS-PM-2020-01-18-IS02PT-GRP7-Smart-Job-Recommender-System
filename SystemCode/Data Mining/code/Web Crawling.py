#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import urllib
import os
import pandas as pd
from selenium import webdriver
from time import sleep
import re
import csv
from selenium.webdriver.common.keys import Keys
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def page_scrap (job_url):
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(executable_path="../code/chromedriver/chromedriver", options=options)
    driver.get(job_url)
    sleep(3)
    html = driver.page_source
    sleep(5)
    driver.close()
    soup = BeautifulSoup(html, 'lxml', from_encoding="utf-8")
    
    job_title = []
    company = []
    job_id = []
    seniority = []
    job_cat = []
    salary = []
    requirements = []
    post_date = []
    emp_type = []
    salary_range = []
    salary_type = []
    experience = []
    

    for entry in soup.find_all('h1', {'id': "job_title"}):
        job_title.append(entry.renderContents())
    for entry in soup.find_all('p', {'class': "f6 fw6 mv0 black-80 mr2 di ttu"}):
        company.append(entry.renderContents())
    for entry in soup.find_all('span', {'class': "black-60 db f6 fw4 mv1"}):
        job_id.append(entry.renderContents())     
    for entry in soup.find_all('p', {'id': "employment_type"}):
        emp_type.append(entry.renderContents())  
    for entry in soup.find_all('p', {'id': "seniority"}):
        seniority.append(entry.renderContents())
    for entry in soup.find_all('p', {'id': "job-categories"}):
        job_cat.append(entry.renderContents()) 
    for entry in soup.find_all('span', {'id': "last_posted_date"}):
        post_date.append(entry.renderContents())       
    for entry in soup.find_all('span', {'class': "salary_range dib f2-5 fw6 black-80"}):
        salary_range.append(entry.renderContents())       
    for entry in soup.find_all('span', {'class': "salary_type dib f5 fw4 black-60 pr1 i pb"}):
        salary_type.append(entry.renderContents())             
    for entry in soup.find_all('p', {'id': "min_experience"}):
        experience.append(entry.renderContents())            
 
        
    salary_all = soup.find_all("span", attrs={'class': 'dib'})
    list_of_inner_text = [x.text for x in salary_all]
    salary_text = '/ '.join(list_of_inner_text)
    salary.append(salary_text)

    requirement_all = soup.find("div", attrs={'id': 'description-content'}).findAll('ul')
    list_of_inner_text = [x.text for x in requirement_all]
    requirements_text = ', '.join(list_of_inner_text)
    requirements.append(requirements_text)    
    

    data = pd.DataFrame(columns = ['Job_Id','Emp_Type','Job_Title','Company','Date_Posted',
                                   'Salary_Range','Salary_Type','Year_Experience',
                                   'Seniority','Category','Requirements'])

    data = data.append({'Job_Id':job_id,'Emp_Type':emp_type,'Job_Title':job_title,'Company':company,
                        'Date_Posted':post_date,'Salary_Range':salary_range,
                        'Salary_Type':salary_type,'Year_Experience':experience,
                        'Seniority':seniority,'Category':job_cat,'Requirements':requirements},ignore_index=True) 

    return data


# In[3]:


def link_scrap(key_word):
    
    output_path = '../job_link/'
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])   
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(executable_path="../code/chromedriver/chromedriver", options=options)
    driver.get("https://www.mycareersfuture.sg/")
    elem = driver.find_element_by_name("search-text")
    sleep(3)
    elem.send_keys(key_word) #job search keyword
    sleep(1)
    elem.send_keys(Keys.RETURN)
    sleep(5)
    current_url = driver.current_url
    first_page = driver.page_source
    sleep(5)
    driver.close()

    #to extract number of job found
    soup = BeautifulSoup(first_page, 'lxml', from_encoding="utf-8")
    job_found = soup.find("div", attrs={'data-cy': 'search-result-headers'})
    text_list = [text for text in job_found.stripped_strings]
    
     
    while True:
        
        try:
            num_job = int(''.join(list(filter(lambda x: x.isdigit(), text_list[0]))))
            print('number of job found:', num_job)
            
            y=0
            links = []
            
            #20 jobs per page
            jobs_per_page = 20
            page_num = (num_job + jobs_per_page-1) // jobs_per_page

            for page in range(page_num):

                new_next_url = current_url.replace('page=' + str(y),'page=' +str(page))
                #print(new_next_url)

                driver2 = webdriver.Chrome(executable_path="../code/chromedriver/chromedriver")
                driver2.get(new_next_url)
                sleep(8)
                html1 = driver2.page_source 
                driver2.close()

                soup = BeautifulSoup(html1, 'lxml', from_encoding="utf-8")

                for entry in soup.find_all("a", attrs={'class': re.compile(r'JobCard')}):
                    link = entry.get('href')
                    links.append(link)

            print('Number of job link extracted:', len(links))

            return links
            
        except ValueError:
            print('Job not found, please try with other keyword')
            
            return 0
        
            break


# In[4]:


def job_crawl(links, key_word):
 
    output_path = '../data_src/'
    job_df = pd.DataFrame()

    for i in range (len(links)):
        job_url  =  "https://www.mycareersfuture.sg%s" % links[i]
        temp_df = page_scrap(job_url)
        job_df = job_df.append(temp_df, ignore_index=True)
        
    job_df.to_csv('{}{}_data.csv'.format(output_path,key_word), index=False)
        
    print('File saved')


# In[25]:


def main():
    
    while True:
        
        print('*************************************************************')
        print('*                                                           *')
        print('*           NUS ISS Group 7 Web-Crawling Program            *')
        print('*                                                           *')
        print('*************************************************************')
        print('\n')
        print('Enter job keyword or type quit to exit')
        print('Please re-run program if encounter chrome session error\n')
        
        enter = str(input('Please enter: ')).lower()
      
        if enter == ('quit' or 'exit'):
            exit()
            break
            
        if enter == ('exit'):
            exit()
            break

        else:
            print('Job keyword entered:', enter)
            jb_links = link_scrap(enter)
            
            if jb_links == 0:
                print('Job not found, please re-run')
                exit()
                
            else:
                job_crawl(jb_links, enter)
                break


# In[26]:


main()


# In[ ]:




