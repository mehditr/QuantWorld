---
title: Credit Card Approval
author: ''
date: '2022-10-19'
slug: credit-card-approval
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2022-10-19T01:39:59+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---






# Credit Card Approval    


Generally, Banks receive lots of requests for loans or credit card however they cannot accept all.
They use many filterings to their default or overdraft risks coming from their customers. Abviously, it is a lenthy process to do that manually however machine learning ppower can help them to reduced their default risks. 
In this notebook I will look at the process of approval credit card based on statistical methond and machine learning approaches


```python
# Import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
sns.set(color_codes=True)
from scipy.stats import chisquare,chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')
# Load dataset
credit = pd.read_csv(r'C:\Users\X550LD\Desktop\ML\credit_clean.csv')

# Inspect data
credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Debt</th>
      <th>Married</th>
      <th>BankCustomer</th>
      <th>Industry</th>
      <th>Ethnicity</th>
      <th>YearsEmployed</th>
      <th>PriorDefault</th>
      <th>Employed</th>
      <th>CreditScore</th>
      <th>DriversLicense</th>
      <th>Citizen</th>
      <th>ZipCode</th>
      <th>Income</th>
      <th>Approved</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>1</td>
      <td>1</td>
      <td>Industrials</td>
      <td>White</td>
      <td>1.25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>ByBirth</td>
      <td>202</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>1</td>
      <td>1</td>
      <td>Materials</td>
      <td>Black</td>
      <td>3.04</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>ByBirth</td>
      <td>43</td>
      <td>560</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>1</td>
      <td>1</td>
      <td>Materials</td>
      <td>Black</td>
      <td>1.50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>ByBirth</td>
      <td>280</td>
      <td>824</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>1</td>
      <td>1</td>
      <td>Industrials</td>
      <td>White</td>
      <td>3.75</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>ByBirth</td>
      <td>100</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>1</td>
      <td>1</td>
      <td>Industrials</td>
      <td>White</td>
      <td>1.71</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>ByOtherMeans</td>
      <td>120</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# 1 - Dealing With Missing Values

In the dataset there are missing values but they have beed replaced with questions mark "?". So for the first action I replace 
the sign with <code>np.NaN</code>, therefore I will deal better with missing values. 
Since there are two general types of data, numerical and categorical, we have to use different methods to impute missing values.
For numerical data, mean or median imputation can be applied as our data is not actually imbalanced, and for categorical data as the mean or median aren't working for them, I would to impute these missing values with the most frequent values as present in the respective columns.
Ignoring the missing values can affect our machine learning model and miss out on information about the dataset that may be useful. The other reason is, some techniques such as Linear Discriminant Analysis (LDA) cannot handle missing values implicilty.


```python

# Replace the '?'s with NaN
credit= credit.replace('?', np.NaN)

# Impute the missing values on numerical based on mean imputation
credit.fillna(credit.mean(), inplace=True)

# Impute missing values applied on categorical data
for col in credit.columns:
    # Check if the column is of object type
    if credit[col].dtypes == 'object':
        # Impute with the most frequent value
        credit = credit.fillna(credit[col].value_counts().index[0])

```

As we have seen, the dataset has different type of variables. Some of them are numerical and the others are categorical.
Age can be an example of numerical type and it is also continous, and Gender is an example of categorical with male and female
that can be masked as 1 and 0. It is important to see what kind of data distinguishes the informative and non-informative. 
It is not applicable to see different types of data, especially for categorical data that can be repeatable, so we can see the component of each columns as the following code :


```python
cat = credit.select_dtypes('object').columns
for col in cat:
    print( col, '--->',credit[col].unique())
```

    Industry ---> ['Industrials' 'Materials' 'CommunicationServices' 'Transport'
     'InformationTechnology' 'Financials' 'Energy' 'Real Estate' 'Utilities'
     'ConsumerDiscretionary' 'Education' 'ConsumerStaples' 'Healthcare'
     'Research']
    Ethnicity ---> ['White' 'Black' 'Asian' 'Latino' 'Other']
    Citizen ---> ['ByBirth' 'ByOtherMeans' 'Temporary']
    


```python

sns.countplot(credit.dtypes.map(str))
plt.show()
# Print DataFrame information
credit_info = credit.info()
print(credit_info)

print('\n')

```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_8_0.png)
    


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Gender          690 non-null    int64  
     1   Age             690 non-null    float64
     2   Debt            690 non-null    float64
     3   Married         690 non-null    int64  
     4   BankCustomer    690 non-null    int64  
     5   Industry        690 non-null    object 
     6   Ethnicity       690 non-null    object 
     7   YearsEmployed   690 non-null    float64
     8   PriorDefault    690 non-null    int64  
     9   Employed        690 non-null    int64  
     10  CreditScore     690 non-null    int64  
     11  DriversLicense  690 non-null    int64  
     12  Citizen         690 non-null    object 
     13  ZipCode         690 non-null    int64  
     14  Income          690 non-null    int64  
     15  Approved        690 non-null    int64  
    dtypes: float64(3), int64(10), object(3)
    memory usage: 86.4+ KB
    None
    
    
    

#  2 - EDA and Statistical Inference
Now, we have a first impression of data. The probable features in a typical credit card application are <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> and finally the <code>ApprovalStatus</code>. This would be a good staring point for us, but we are not still aware of the importance of features. At the first step, I will ignore some features like <code>DriversLicense</code> and <code>ZipCode</code> as they are not informative as the other features in the dataset for predicting credit card approvals. To get a better sense, we can apply some hypothesis testing along with their visualizations to see what features are probably more important the others


```python
plt.figure(figsize=(20,7),dpi=300)

sns.countplot(data=credit,x='Industry',hue='Approved')
plt.title('Figure_1 - Approved and Not Approved Based on Industry')

plt.tight_layout()
```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_10_0.png)
    



```python
Perc_App_Ind = pd.DataFrame(round(credit.loc[:,['Approved','Industry']].groupby('Industry').sum()/
                                  credit.loc[:,['Approved','Industry']].groupby('Industry').count(),2))
Perc_App_Ind.sort_values('Approved',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Approved</th>
    </tr>
    <tr>
      <th>Industry</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utilities</th>
      <td>0.84</td>
    </tr>
    <tr>
      <th>InformationTechnology</th>
      <td>0.71</td>
    </tr>
    <tr>
      <th>Transport</th>
      <td>0.67</td>
    </tr>
    <tr>
      <th>Materials</th>
      <td>0.65</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.56</td>
    </tr>
    <tr>
      <th>Industrials</th>
      <td>0.52</td>
    </tr>
    <tr>
      <th>Energy</th>
      <td>0.45</td>
    </tr>
    <tr>
      <th>CommunicationServices</th>
      <td>0.42</td>
    </tr>
    <tr>
      <th>ConsumerStaples</th>
      <td>0.35</td>
    </tr>
    <tr>
      <th>Research</th>
      <td>0.30</td>
    </tr>
    <tr>
      <th>Financials</th>
      <td>0.27</td>
    </tr>
    <tr>
      <th>ConsumerDiscretionary</th>
      <td>0.24</td>
    </tr>
    <tr>
      <th>Real Estate</th>
      <td>0.23</td>
    </tr>
    <tr>
      <th>Healthcare</th>
      <td>0.13</td>
    </tr>
  </tbody>
</table>
</div>



So,'Figure_1' shows which industries people are coming from, and they have been approved or not approved.
So basically some industries such as <code>Healthcare</code> or <code>Consumer Discretionary</code> are more failed.
Therefore that <code>Industry</code> can be a potential feature as an important one.


```python
sns.displot(data=credit,x='Age',bins=30,kde=True,hue='Approved')
plt.annotate('Two Tails cross', xy=(55, 10), xytext=(60,23),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.title (' Figure_2 : Approved and UnApproved base on Age Distribution')
plt.show()
```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_13_0.png)
    


The second graph is an interesting one. In general, there are many reasons why a bank will reject applications when a person has a low or moderate income or does not have enough funds to support themselves against debts, and this happens more often in the early years of work. . Therefore, according to the graph, it is possible to see the ages below 40 years (almost or slightly less than 40 years), the number of unapproved applications is more than one approved one.
After 40 (or under 40) the number of endorsements increases, which tells us that age is an informative characteristic.


```python
plt.figure(figsize=(4,4))

plt.pie(credit['Approved'].value_counts(),colors=['g','pink']
        ,explode=[0.03,0.03],autopct='%.2f',shadow=5,radius=1.1)

plt.tight_layout()
plt.title('Figure_3 : Approx. Balanced Data')
plt.show()
print(pd.crosstab(credit.Approved, credit.Approved))
```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_15_0.png)
    


    Approved    0    1
    Approved          
    0         383    0
    1           0  307
    

Figure_3: As I mentioned earlier, our data set is not imbalanced. In general, data sets for predicting default rates or detecting fraud are imbalanced data. Working with imbalanced data requires techniques to obtain meaningful output and is beyond the scope of this project. As we can see here, the number of approved credit cards and 307 unapproved credit cards is 383.


```python
sig_Item =credit.groupby('Approved')['Age','Debt','CreditScore','Income'].mean()
pd.DataFrame(sig_Item)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Debt</th>
      <th>CreditScore</th>
      <th>Income</th>
    </tr>
    <tr>
      <th>Approved</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29.773029</td>
      <td>3.839948</td>
      <td>0.631854</td>
      <td>198.605744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33.686221</td>
      <td>5.904951</td>
      <td>4.605863</td>
      <td>2038.859935</td>
    </tr>
  </tbody>
</table>
</div>



In the table above, we can see some continuous features separated by the "Approved" column. Except for age, all have significant differences in their means. Particularly, there are noticeable large gaps in average income and credit scores. So here it is good to check their graphs and see their distribution.


```python
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
sns.boxplot(data=credit,x='Approved',y='Age')

plt.subplot(2,2,2)
sns.boxplot(data=credit,x='Approved',y='Debt')

plt.subplot(2,2,3)
sns.boxplot(data=credit,x='Approved',y='CreditScore')

plt.subplot(2,2,4)
sns.boxplot(data=credit,x='Approved',y='Income')

plt.tight_layout()
plt.show()
```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_19_0.png)
    


The second row graphs demonstrate some weird points that can be outliers. Let's try to see what is the characteristics of those two highest points in credit score and income


```python
print('The highest point in Credit Score :')
print('\n')
maxim_1 = credit['CreditScore'].idxmax()
print(credit[['Age','Debt','CreditScore','Income','Industry']].iloc[maxim_1])

print('------------------------------------')

print('The highest point in Income :')
print('\n')
maxim_2 = credit['Income'].idxmax()
print( credit[['Age','Debt','CreditScore','Income','Industry']].iloc[maxim_2])

```

    The highest point in Credit Score :
    
    
    Age                            25.67
    Debt                            12.5
    CreditScore                       67
    Income                           258
    Industry       InformationTechnology
    Name: 121, dtype: object
    ------------------------------------
    The highest point in Income :
    
    
    Age                  17.5
    Debt                 22.0
    CreditScore             0
    Income             100000
    Industry       Healthcare
    Name: 317, dtype: object
    


```python
maxim_2 = credit['Income'].idxmax()
credit[['Age','Debt','CreditScore','Income','Industry']].iloc[maxim_2]
```




    Age                  17.5
    Debt                 22.0
    CreditScore             0
    Income             100000
    Industry       Healthcare
    Name: 317, dtype: object



The interesting thing to note here is that the highest credit score is related to the IT industry, which ranks second in "approved" programs. However, the highest revenue comes from the healthcare industry, which is the last industry to receive an approved application and both have the same amount of debt. Is there bias in these decisions?!!!!!!
That can be a trigger for data scientists !!!


```python
Customer = credit[credit['BankCustomer']==1]
Non_Customer = credit[credit['BankCustomer']==0]

g = sns.FacetGrid(credit, col="Approved",  row="BankCustomer")
g.map_dataframe(sns.histplot, x="Debt")
plt.annotate('More Debt', xy=(20, 5), xytext=(30,23),
            arrowprops=dict(facecolor='black', shrink=0.05))
g.fig.subplots_adjust(top=0.85) 
g.fig.suptitle('Figure_5')
plt.show()
```


    
![png](Credit_Approval_Final_files/Credit_Approval_Final_24_0.png)
    


Figure_5 : There is an argue that some bank customers who are approved by the bank have more debt than others who are not their customers. In the figure above, I have split the data based on the "Bank Customer" and "Verified" attributes.
This graph shows that bank customers have more debts than those who are not, and this difference is significant.

# 3 - Machine Learning Techniques

In this section, I will use two techniques Logistics regression and Support Vector Machines for classifying the data.

**3-1 : Logistic Regression**


```python
# removing non-informative columns
df = credit.copy()
df = df.drop(['DriversLicense','ZipCode'],axis=1)
```


```python
df = pd.get_dummies(df)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Debt</th>
      <th>Married</th>
      <th>BankCustomer</th>
      <th>YearsEmployed</th>
      <th>PriorDefault</th>
      <th>Employed</th>
      <th>CreditScore</th>
      <th>Income</th>
      <th>...</th>
      <th>Industry_Transport</th>
      <th>Industry_Utilities</th>
      <th>Ethnicity_Asian</th>
      <th>Ethnicity_Black</th>
      <th>Ethnicity_Latino</th>
      <th>Ethnicity_Other</th>
      <th>Ethnicity_White</th>
      <th>Citizen_ByBirth</th>
      <th>Citizen_ByOtherMeans</th>
      <th>Citizen_Temporary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>1</td>
      <td>1</td>
      <td>1.25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>1</td>
      <td>1</td>
      <td>3.04</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>560</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>1</td>
      <td>1</td>
      <td>1.50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>824</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>1</td>
      <td>1</td>
      <td>3.75</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>1</td>
      <td>1</td>
      <td>1.71</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>



Since there are a number of categorical variables, I use a method called "One Hot Encode" which is suitable when there is no relationship between the categorical variables. For example, we can see that the industry attribute is divided by industry names and their values are 0 and 1. Since our goal is to use models to get the right classification and predict new customers, we need to scale our data, so the mentioned method should be used.


```python
from sklearn.preprocessing import MinMaxScaler

```


```python
scaler= MinMaxScaler(feature_range=(0,1))
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,columns=df.columns)
scaled_df['Approved']=df['Approved']
```


```python
y= scaled_df['Approved']
X= scaled_df.drop('Approved',axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=1000)

```


```python
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train,y_train)

```




    LogisticRegression()




```python
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score,precision_score

y_pred = log_model.predict(X_test)
y_pred_train = log_model.predict(X_train)
print('Accuracy of logistic model is for Train \t:', round(log_model.score(X_train,y_train ),3))
print('Accuracy of logistic model is for Test \t\t:', round(log_model.score(X_test,y_test),3))
#log_model.score
print('Recall of logistic model is for Test \t\t:', round(recall_score(y_test,y_pred),3))
print('f1_score of logistic model is for Test \t\t:', round(f1_score(y_test,y_pred),3))
print('Precision of logistic model is for Test \t:', round(precision_score(y_test,y_pred),3))

confusion_matrix(y_test,y_pred)

```

    Accuracy of logistic model is for Train 	: 0.87
    Accuracy of logistic model is for Test 		: 0.868
    Recall of logistic model is for Test 		: 0.871
    f1_score of logistic model is for Test 		: 0.871
    Precision of logistic model is for Test 	: 0.871
    




    array([[ 97,  15],
           [ 15, 101]], dtype=int64)




```python
from sklearn.model_selection import GridSearchCV

tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

param_grid = dict(tol=tol,max_iter = max_iter)
```


```python
grid_model = GridSearchCV(estimator = log_model, param_grid = param_grid ,cv= 5)

grid_model_output = grid_model.fit(X_train,y_train)

best_score, best_params = grid_model_output.best_score_ , grid_model_output.best_params_

best_model = grid_model_output.best_estimator_
print(" The Accuracy is for Test is \t\t:", best_model.score(X_test,y_test))


```

     The Accuracy is for Test is 		: 0.868421052631579
    

I have run the logistic regression and for the first part I have reach good result for both train and test about 87%. This shows that almost the model is not overfitted since both the accuracy of train and test are close to each other. 
For getting the better result, I have applied a method call 'GridSearchCV'. Generally, Grid search procedure is used for tuning hyperparameters and fit the learner on train and validation ( or test) sets repeatedly and in the end find a best pramaeter for the model. 
After applying this method on and trying ti get the result with best parameters, I have reach again the accuracy of test as the same as previous one.

 **3-2 Support Vector Machine**


```python
from sklearn.svm import SVC
```


```python
SVM_model = SVC()
SVM_model.fit(X_train,y_train)
```




    SVC()




```python
y_hat = SVM_model.predict(X_test)
```


```python
prediction = pd.DataFrame({'y_test':y_test,'y_hat':y_hat})
prediction
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>y_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>389</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>609</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>393</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>338</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>353</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>267</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>638</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>228 rows × 2 columns</p>
</div>




```python
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, confusion_matrix


def SVM_report(X_train,y_train,X_test,y_test,C=1,gamma='scale',kernel='rbf',class_weight = None):
    svc=SVC(C=C,gamma=gamma,kernel=kernel,class_weight = class_weight)
    svc.fit(X_train,y_train)
    y_hat=svc.predict(X_test)
    
    cm = confusion_matrix(y_test,y_hat)
    accuracy = round(accuracy_score(y_test,y_hat),4)
    error_rate= round((1-accuracy),4)
    precision = round(precision_score(y_test,y_hat),2)
    recall = round(recall_score(y_test,y_hat),2)
    f1score = round(f1_score(y_test,y_hat),2)
    cm_labled = pd.DataFrame(cm, index=['Actual : negative','Actual : positive'],columns=['Predict : negative','Predict: positive'])
    
    print('The metrics are as follow :')
    
    print('Accuracy = {}'.format(accuracy))
    print('error_rate = {}'.format(error_rate))
    print('precision = {}'.format(precision))
    print('recall = {}'.format(recall))
    print('f1score = {}'.format(f1score))
    
    return cm_labled
```


```python
SVM_report(X_train,y_train,X_test,y_test,kernel='rbf')
```

    The metrics are as follow :
    Accuracy = 0.8816
    error_rate = 0.1184
    precision = 0.87
    recall = 0.9
    f1score = 0.89
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predict : negative</th>
      <th>Predict: positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual : negative</th>
      <td>97</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Actual : positive</th>
      <td>12</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>



We can see an improvment in all metrics regarding to logistic regression in the first step. Now I would like to tune the hyperparameters in SVM. For this reason, I will apply an approach call kernel tricks. As the shape of logistic regression is 'S' shape and the function of logistic regression is 'Sigmoid', I will implement another kernels such as ' Radial Basis Function'( or'rbf') and 'Polynomial' (or 'poly') for tuning.

# Tuning Hyperparameters


```python
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, confusion_matrix,make_scorer
my_param_grid ={'C':[10,100,1000],'gamma':['scale',0.01,0.001], 'kernel':['rbf','poly']}
f1=make_scorer(f1_score)
```


```python
grid = GridSearchCV(estimator = SVC(),param_grid = my_param_grid, refit=True,verbose=2, cv=5, scoring=f1)
```


```python
grid.fit(X_train,y_train)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.1s
    [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END .....................C=10, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ......................C=10, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ......................C=10, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END .....................C=10, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ....................C=100, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ....................C=100, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=scale, kernel=rbf; total time=   0.0s
    [CV] END ...................C=1000, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=scale, kernel=poly; total time=   0.0s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.01, kernel=poly; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ....................C=1000, gamma=0.001, kernel=rbf; total time=   0.0s
    [CV] END ...................C=1000, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=0.001, kernel=poly; total time=   0.0s
    [CV] END ...................C=1000, gamma=0.001, kernel=poly; total time=   0.0s
    




    GridSearchCV(cv=5, estimator=SVC(),
                 param_grid={'C': [10, 100, 1000], 'gamma': ['scale', 0.01, 0.001],
                             'kernel': ['rbf', 'poly']},
                 scoring=make_scorer(f1_score), verbose=2)



Since we have 3 states of <code>'C'</code> and 3 states of <code>'gamma'</code> and 2 states of <code>'kernel'</code> and 5 cross validation, the total fiiting will be 90 fit cases


```python
grid.best_params_
```




    {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}




```python
grid.best_estimator_
```




    SVC(C=10, gamma=0.01)




```python
y_hat_optimized = grid.predict(X_test)
```


```python
prediction['y_hat_optimized']=y_hat_optimized
prediction.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>y_hat</th>
      <th>y_hat_optimized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>565</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>218</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>157</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>431</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>294</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>588</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>527</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>159</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>368</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>329</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>273</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>353</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>267</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>638</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
SVM_report(X_train,y_train,X_test,y_test,C=10,gamma=0.01,kernel='rbf')
```

    The metrics are as follow :
    Accuracy = 0.8772
    error_rate = 0.1228
    precision = 0.84
    recall = 0.93
    f1score = 0.89
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predict : negative</th>
      <th>Predict: positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual : negative</th>
      <td>92</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Actual : positive</th>
      <td>8</td>
      <td>108</td>
    </tr>
  </tbody>
</table>
</div>



In the last step for SVM, we see an improvment in <code>'recall'</code> but a decrease in <code>'Accuracy'</code>. Sometimes in classification problems, data scientists look at the <code>'f1_score'</code> and then <code>'recall'</code>. For those who consider the weight of <code>'f1_score'</code> and <code>'recall'</code> more in classification, this result might be valuable.
