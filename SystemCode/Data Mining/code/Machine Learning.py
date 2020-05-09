#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
import glob, os
import warnings
warnings.filterwarnings("ignore")


# In[26]:


import statsmodels.formula.api as sm
import patsy
import itertools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GridSearchCV,learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from sklearn.feature_extraction import stop_words
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot_ng as pydot
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from string import printable


# In[28]:


def data_ingestion(src_path, extension, output_path):
    
    all_filenames = [i for i in glob.glob('{}*.{}'.format(src_path,extension))] 
    print('Source file(s) loaded:', all_filenames)
    
    #combine all files in the list 

    df_raw = pd.concat([pd.read_csv(f, encoding='unicode escape',skiprows=0) for f in all_filenames ]) 
    df_raw.reset_index(inplace=True) 
    df_raw = df_raw.drop(columns=['index','Unnamed: 0']) 
    
    return df_raw


# In[29]:


def data_clean(df,title_key,drop_title_key,drop_cat_key):
    
    #remove duplicate based on Job ID
    job_clean = df.drop_duplicates(subset='Job_Id', keep='first')
    #drop Salary_Type due to only one unique value 'Monthly'
    job_clean = job_clean.drop(columns='Salary_Type')
    #remove job without title
    job_no_title = job_clean['Job_Title'] == ''
    job_clean = job_clean[~job_no_title]
    #remove row with all NaN value
    job_clean[job_clean.isnull().any(axis=1)]
    job_clean = job_clean.dropna()
    
    #perform data cleaning on every row and columms
    clean_list = "(\[|\]|b'|Requirements|'|amp;|xa0|\\\|xe2x80x93|\\n|div class=|div class=|span class=|dib|lh-solid|/span|f5-5 i fw4 gray|f5 fw4 ph1|<|>|/div|\")"
    for col in job_clean.columns.difference(['Requirements']):
        job_clean[col]=job_clean[col].str.replace(clean_list, "")

    #space remain for Requirements column    
    job_clean['Requirements']=job_clean['Requirements'].str.replace(clean_list, " ")

    #remove all non-ascii char except punctuation, digits, ascii_letters and whitespace
    job_clean['Requirements'] = job_clean['Requirements'].apply(lambda y: ''.join(filter(lambda x: x in printable, y)))
    
    #further remove job with same data from all columns
    job_clean = job_clean.drop_duplicates(subset=job_clean.columns, keep='first') 
        
    #further filter on job title with specific keywords

    key = '|'.join(title_key)
    data_job = job_clean['Job_Title'].str.upper().str.contains(key)
    job_clean = job_clean[data_job]

    #remove job title with unwanted keywords
    
    key2 = '|'.join(drop_title_key)
    non_data_job = job_clean['Job_Title'].str.upper().str.contains(key2)
    job_clean = job_clean[~non_data_job]
    
    #remove job with multiple category
    cat_list = "(/|and)"
    job_clean['Category']=job_clean['Category'].str.replace(cat_list, ",")
    job_clean['Cat_num'] = job_clean['Category'].str.count(',')
    
    multiple_cat = job_clean['Cat_num']>5
    job_clean = job_clean[~multiple_cat]
    job_clean = job_clean.drop(columns='Cat_num')
    
    #remove job with no or multiple seniority
    senior_rule = (job_clean['Seniority'].str.count(',')>=1) | (job_clean['Seniority']=='')
    job_clean = job_clean[~senior_rule]
   
    #remove job cat with specific keywords

    key3 = '|'.join(drop_cat_key)
    rare_cat = job_clean['Category'].str.upper().str.contains(key3)
    job_clean = job_clean[~rare_cat]
    
    #remove row without salary
    no_salary = job_clean['Salary_Range'].str.contains('Salary undisclosed')
    df_salary = job_clean[~no_salary]
    df_no_salary = job_clean[no_salary]
    df_salary = df_salary.reset_index(drop=True)
    
    req_empty = []

    for i in range (len(df_salary)):
    
        if((len(df_salary['Requirements'][i]))<5):
            req_empty.append(i)
           
    #clean & remove row without requirements
    df_salary['Requirements']=df_salary['Requirements'].str.replace('(\n)', "")
    df_salary = df_salary.drop(req_empty)
    df_salary = df_salary.reset_index(drop=True)

    return df_salary


# In[30]:


#define list of filters

title_key = ['DATA', 'MACHINE','ANALYST','MACHINE LEARNING','ANALYTICS', "SCIENCE", '4.0','APPLICATION', 'DEEP LEARNING',
             'RESEARCH','NLP', 'ARTIFICIAL', "INTELLIGENT", 'AI', 'SCIENTIST','SYSTEM','Industry', 'IOT', 'FINANCE', 'FINTECH', 
             'SOFTWARE', 'ENGINEER', 'ENGINEERING','PROFESSOR','BUSINESS', 'DEVELOPER', 'INDUSTRIAL','AUTOMATION', 'CLOUD',
             'SOLUTION','ARCHITECT', 'MANAGER','VP','PRESIDENT', 'TECHNOLOGY', 'SPECIALIST', 'TECHNICAL','LEAD','TECHNOLOGIST']

unwanted_title_key = ['PHYSIOTHERAPIST','ACCOUNT','AUDIT','COUNSEL','EXECUTIVE','SALES','GENERAL','MARKET','ELECTRICAL','BUSINESS',
                      'ADMIN','CUSTOMER','OFFICER','OPERATION', 'MECHANICAL','CHEMICAL','COORDINATOR','LECTURER','TECHNICIAN']

unwanted_cat_key = ['HUMAN','SOCIAL','THERAPY','TAXATION','CUSTOMER','INTERIOR', 'ADMIN','BUILDING', 'SECRETARIAL','INVESTIGATION', 
                'AUDITING', 'ENVIRONMENT','SALES', 'MARKETING','ADVERTISING','CONSTRUCTION', 'DESIGN','LEGAL','HOSPITALITY',
                'PROFESSIONAL']

cat_list = ['Information Technology', 'Telecommunications', 'Engineering','Sciences', 'Finance','Healthcare','Management',
            'Consulting','Logistics', 'Civil', 'Others']

edu_list = ['phd','doctor','master','degree','computer science','engineering','statistic','math','computer engineering',
            'business','ph.d']

skills_list = ['python','java','scala','hadoop','sql','spark','tensorflow','scikit','linux','pytorch','theano','caffe','Matlab',
               'perl','deep learning','nlp','apache','mapreduce','aws','azure','container','kafka','cassandra', 'c\++' ,'julia',
               'jupyter','nltk','tableau','power bi','sas','pandas','git','hive','impala','agile','machine learning','bash',
               'natural language','oracle','cloud','flask','golang','optimization','c#','opencv','computer vision','api','jira',
               'unix','bash','docker','keras', 'qlik','gcp','scrum', 'airflow','.net','d3.js']


# In[31]:


def salary_feature(df, low, med, high):
    
    #extract salary columns due to contain multiple information
    salary_range = df["Salary_Range"].str.split("to", n = 2, expand = True) 

    #Give columns name to the dataframe
    salary_range = salary_range.rename({0:'Min_Salary',1:'Max_Salary'}, axis='columns')

    #removed $ and , from salary 
    for col in salary_range.columns:
        salary_range[col]=salary_range[col].str.replace('(\$|,)', '')

    #convert from ojbect to float for statistical infomation
    salary_range['Min_Salary'] = salary_range['Min_Salary'].astype('float64')
    salary_range['Max_Salary'] = salary_range['Max_Salary'].astype('float64')
    
    #concat min_max salary dataframe with salary range dataframe
    df_salary1 = pd.concat([df, salary_range], axis=1)
    df_salary1 = df_salary1.drop(columns='Salary_Range')  
    
    #create a condition to check for high outliers
    abovemean_min = round(10*np.mean(df_salary1['Min_Salary']),0)
    abovemean_max = round(10*np.mean(df_salary1['Max_Salary']),0)
    
    #convert yearly salary into monthly salary

    df_salary1['Min_Salary'] = np.where((df_salary1['Min_Salary'] > abovemean_min),
                                    round((df_salary1['Min_Salary']/12),0), df_salary1['Min_Salary'])

    df_salary1['Max_Salary'] = np.where((df_salary1['Max_Salary'] > abovemean_min),
                                    round((df_salary1['Max_Salary']/12),0), df_salary1['Max_Salary'])
    
    #drop unrealistic min and max monthly salary range (which is more than 10 times)
    min_max_abnormal = (df_salary1['Max_Salary']>10*df_salary1['Min_Salary'])
    df_salary1 = df_salary1[~min_max_abnormal]
    
    #drop job with max salary less than 2500, assuming data entry/admin/operator job
    low_sal = ((df_salary1['Min_Salary']<=1800) | (df_salary1['Max_Salary']<=2500))
    df_salary1 = df_salary1[~low_sal]
    
    #create new feature for average salary
    df_salary1['Avg_Salary'] = (df_salary1['Min_Salary'] + df_salary1['Max_Salary']) / 2
    
    #drop job with outlier salary
    salary_outlier = ((df_salary1['Avg_Salary']>20000) | (df_salary1['Avg_Salary']<3000))
    df_salary1 = df_salary1[~salary_outlier]
    
    #bin salary into 3 groups:
    #3000 to 4500 - Low
    #4500 to 6000 - Med
    #6000 and above - High

    bins = [low, med, high, np.inf]
    names = ['Low', 'Med', 'High']

    df_salary1['Salary_range'] = pd.cut(df_salary1['Avg_Salary'], bins, labels=names)
    df_salary1 = df_salary1.reset_index(drop=True)
    
    return df_salary1


# In[32]:


def emp_type(df):
    
    #remove others employment type
    type_key = ['PART TIME','TEMPORARY','INTERNSHIP','FLEXI','FREELANCE']
    key = '|'.join(type_key)
    non_type = df['Emp_Type'].str.upper().str.contains(key)
    df = df[~non_type]
    df = df.reset_index(drop=True)

    #consolidate employment type
    consolidate = "(Full Time|Permanent, Full Time)"
    df['Emp_Type']=df['Emp_Type'].str.replace(consolidate, "Permanent")

    consolidate = "(Contract, Full Time)"
    df['Emp_Type']=df['Emp_Type'].str.replace(consolidate, "Contract")

    consolidate = "(Contract, Permanent, Full Time)"
    df['Emp_Type']=df['Emp_Type'].str.replace(consolidate, "Cont_Perm")
    
    return df


# In[33]:


def cat_name(df):
    
    stacked = pd.DataFrame(df['Category'].str.split(',').tolist()).stack()
    cat_count = pd.DataFrame(stacked.value_counts(), columns=['Count']).reset_index()
    cat_count1 = []

    for i in range (len(cat_count)):
        cat_count1.append(cat_count['index'][i].lstrip())
    
    cat_name = list(dict.fromkeys(cat_count1))
    
    return cat_name


# In[34]:


def clean_kie(df):

    #extract only number from string
    df['Year_Experience'] = df['Year_Experience'].str.extract('(\d+)')
    
    #remove comma from cell with string
    clean_list = "(,|;|â||¦|®|)"
    for col in df.columns.difference(['Year_Experience','Min_Salary','Max_Salary','Avg_Salary']):
        df[col]=df[col].str.replace(clean_list, "")
        
    #remove extra whitespace between string
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #remove row without year of experience
    df = df.dropna(subset=['Year_Experience']).reset_index(drop=True)
    
    #fill NaN in year of experience with 0
    #df['Year_Experience'] = df['Year_Experience'].fillna(0)
    
    return df


# In[35]:


def job_edu_cat(df, cat_list, edu_list):
    
    for cat in cat_list:
        df[cat] = np.where(df['Category'].str.lower().str.contains(cat),1,0)
        
    for edu in edu_list:
        df[edu] = np.where(df['Requirements'].str.lower().str.contains(edu),1,0)

    return df


# In[36]:


def skill_cat(df, skills_list):
    
    for skill in skills_list:
        df[skill] = np.where(df['Requirements'].str.lower().str.contains(skill),1,0)         

    return df


# In[37]:


#word count function
def word_count(df_col):

    str_counts = 0
    sum_str = 0

    for i in range (len(df_col)):    
        str_counts = len(df_col[i].split())
        sum_str = sum_str + str_counts

    print(sum_str)
    
#frequent word function
def freq_words(word_count, features):

    num_word = np.asarray(word_count.sum(axis=0)).reshape(-1)
    most_count = num_word.argsort()[::-1]
    key_word = pd.Series(num_word[most_count], 
                           index=features[most_count])

    return key_word


# In[38]:


def stop_word_fil(df):
    
    #stop words were added to filter some generic recurring business terms.
    stop = stopwords.words('english')
    stop += ['regret','shortlisted', 'candidates','notified','etc', 'take', 'hands','added','able','writting',
             'year','years','least', 'related','using', 'and', 'ability','work','skills','advantage','written'
            'develop','good','team','design','knowledge','experience','following','areas', 'ability','and','in','to']
    
    #most common words for requirements
    cvt = CountVectorizer(lowercase=True, strip_accents='unicode',max_features=5000, min_df=1, max_df=0.9,
                          stop_words=stop, ngram_range=(1,2))
    vect_word = cvt.fit_transform(df['Requirements'])
    features = np.array(cvt.get_feature_names()) 

    key_word = freq_words(vect_word, features)
    
    #update stop_word with common words
    new_stop = key_word[key_word<5].index
    stop.extend(new_stop)
    
    #number of word found in Requirements column before clean
    print('Number of words before filter:')
    word_count(df['Requirements'])
    
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    df['Requirements'] = df['Requirements'].str.replace(pat, " ")
    df['Requirements'] = df['Requirements'].map(lambda x: x.strip())
    df['Requirements'] = df['Requirements'].replace({' +':" "},regex=True)
    
    print('Number of words after filter:')
    word_count(df['Requirements'])
    
    return df


# In[39]:


def pred_target(data_df):
    
    #Dummified Seniority columns to use as predictor features
    seniority_cat=data_df['Seniority'].str.get_dummies()
    emp_cat=data_df['Emp_Type'].str.get_dummies()
    df = pd.concat([data_df, seniority_cat, emp_cat], axis=1)

    #CountVectorizer job requirements columns

    stop_ml = stopwords.words('english')

    stop_ml += ['possess','work','back','ability','communication', 'participate', 'like', 'tools','distributed',
                'contribute','proven','engage','understanding','excellent', 'teams','experienced', 'familiarity',
                'partners', 'study', 'well','preferably','user','field','experience','english', 'level','sets','stakeholders',
                'delivery','implementation','relevant','state','exposure','record','problems','define','open','subject',
                'proficient','understand','proficiency','efficiency','build', 'ideas','basic', 'technical','life','help']

    cvec = CountVectorizer(lowercase=True, strip_accents='unicode',
                           max_features=500, min_df=10, max_df=0.6, 
                           stop_words=stop_ml,ngram_range=(1,2))
    cvec.fit(df['Requirements'])

    #creating predictor and target dataset
    model_data = df.drop(columns=['Job_Title','Company','Seniority','Category','Min_Salary',
                                   'Max_Salary','Emp_Type','Avg_Salary', 'Job_Id', 'Date_Posted'])

    nlp = pd.DataFrame(cvec.transform(model_data['Requirements']).todense(),columns=cvec.get_feature_names())

    senior_nlp = pd.concat([model_data, nlp], axis=1)

    X = senior_nlp.drop(columns=['Salary_range','Requirements'])
    y = senior_nlp['Salary_range'].values

    return X, y


# In[40]:


def model_train(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    dtc = DecisionTreeClassifier(max_depth=5, random_state=42)
    dtc_nlp = dtc.fit(X , y)
    
    return dtc_nlp


# In[54]:


def print_report(model, X, y, output_path):
    
    print('classficication report: \n')
    print(classification_report(y,model.predict(X),target_names=["Low", "Med", "High"]))
    print('\n')

    print('confusion matrix: \n')
    print(pd.DataFrame(confusion_matrix(y,model.predict(X)),
                 index=['Actual Low','Actual Med', 'Actual High'],
                 columns=['Pred Low','Pred Med','Pred High']))
    print('\n')

    features = np.array(X.columns)
    dt_coefs = pd.DataFrame({'coef':model.feature_importances_, 'abs coef':abs(model.feature_importances_)},index=features)
    dt_coefs = dt_coefs.sort_values('coef',ascending=False)
    dt_coefs.head(15)
    print('Feature Importances\n')
    print(dt_coefs.head(14))
    print('\n')
    
    #this function is use to export decision tree rules
    tree_rules = export_text(model, feature_names=list(X))
    
    with open("{}Tree_rules.txt".format(output_path), "w") as text_file:
        print(tree_rules, file=text_file)
    
    print('Decision tree rules: ')
    print(tree_rules)
    print('\n')
    print('Tree_rules.txt saved')


# In[51]:


#this function is use to print Decision Tree into pdf file
def output_pdf(model, X_nlp):

    #add graphviz to path
    os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

    feature_names = X_nlp.columns
    class_name = dtc_nlp.classes_.astype(str)
    
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data,
                         feature_names=feature_names,
                         class_names=class_name,
                         filled=True, rounded=True,
                         special_characters=True,
                          node_ids=1,)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("{}Decision_Tree.pdf".format(output_path))
    print('Decision Tree diagram saved')


# In[55]:


def main():
    
    src_path = '../data_src/' #web-crawled data source path
    extension = 'csv' #data source format
    output_path = '../output/' #database output path

    #define salary range
    low = 3000
    med = 4500
    high = 6000
    
    #data ingestion and data cleaning

    df_raw = data_ingestion(src_path, extension, output_path)
    df = data_clean(df_raw, title_key, unwanted_title_key, unwanted_cat_key)

    #feature engineering
    df1 = salary_feature(df, low, med, high)
    df1 = emp_type(df1)
    df1 = clean_kie(df1)
    df1 = job_edu_cat(df1, cat_list, edu_list)
    df1 = skill_cat(df1, skills_list)
    data_df = df1.copy()
    #data_df = stop_word_fil(df1)    
    
    X, y = pred_target(data_df)
    model = model_train(X, y)
    print_report(model, X, y, output_path)


# In[53]:


main()


# In[ ]:




