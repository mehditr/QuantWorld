---
title: Statistical Inference - Part I
author: ''
date: '2023-01-03'
slug: statistical-inference
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2023-01-03T14:55:33+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


The goal of ***Explanatory Data Analysis*** is to extract some information and knowledge about dataset and its components especially before going through a deep data analysis or machine learning parts. One the primary steps in this way is, ***Statistical Inference*** that leads us to discuss and create the meaningful questions about the variables.

This is the fact that machine learning helps scientists to create a prediction models (in supervised learning ) and decsribe the data structure and finding specific patterns (in unsupervised learning), however, it is invetible that machine learning tells us nothing about parameter estimation or and specific answers for some univariate analysis. 
As statistics is vital steps for data modeling and machine learning also, many scientists are trying to find some relations in univariate analysis. Fore instance it important for them to know is ***Education*** statistically a relevant variable to explain the default rate?  
Such questions lead us to create ***hypothesis testing*** examination to find out significant variables in the univariate analysis. 

In the recent post about [Credit Approval](https://quantworld.netlify.app/post/credit-card-approval/), I intended to explain more about machine learning techniques and tuning the hyperparameters and also some ***EDA*** at the first of the post, but now I would like to consider some hypothesis testing about the credit approval dataset and see how we can extract some informative knowledge variables, specially when the variables are skewed. This might happen more in unbalanced dataset.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```


```python
credit = pd.read_csv(r'C:\Users\X550LD\Desktop\ML\credit_clean.csv')
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



One the ***EDA*** done in the Credit Aprroval investigation, there was a claim that [***bank customer has more debt that non-bank customer***](https://quantworld.netlify.app/img/Credit_Approval_Final_24_0.png). This statement might be true or not for gender variable. So, for having more consideration about these statements, we will create two hypothesis testings regarding being customer of the bank and Gender in respect to Debt which they might be important for getting the approval for credit card.

So, first of all, we will see both distributions are right skewed 


```python
# splitting data  
Yes_Customer= credit[credit.BankCustomer== 1].Debt
No_Customer = credit[credit.BankCustomer==0].Debt

fig = plt.figure(figsize=(14,7))
n, bins, patches = plt.hist(Yes_Customer, bins =50, facecolor='blue', alpha=0.5,label='Yes_Customer')
n, bins, patches = plt.hist(No_Customer, bins =50,facecolor='red', alpha=0.5,label='No_Customer')
plt.axvline(Yes_Customer.mean(),linestyle='--',color='blue',)
plt.axvline(No_Customer.mean(),linestyle='--',color='red',)
plt.xlabel('Debt')
plt.legend();
```


<img src="/img/Statistical_Inference_2_8_0.png" alt="" />     

    



```python
# splitting data  
Male= credit[credit.Gender== 1].Debt
Female = credit[credit.Gender==0].Debt

fig = plt.figure(figsize=(14,7))
n, bins, patches = plt.hist(Male, bins =50, facecolor='blue', alpha=0.5,label='Male')
n, bins, patches = plt.hist(Female, bins =50,facecolor='red', alpha=0.5,label='Female')
plt.axvline(Male.mean(),linestyle='--',color='blue',)
plt.axvline(Female.mean(),linestyle='--',color='red',)
plt.xlabel('Debt')
plt.legend();
```


 <img src="/img/Statistical_Inference_2_9_0.png" alt="" />        

    



```python
BankCustomer_diff = credit.groupby('BankCustomer')['Age','Debt','YearsEmployed','Income'].mean()
BankCustomer_diff
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
      <th>YearsEmployed</th>
      <th>Income</th>
    </tr>
    <tr>
      <th>BankCustomer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29.394233</td>
      <td>4.009325</td>
      <td>1.766994</td>
      <td>481.226994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32.169791</td>
      <td>4.990512</td>
      <td>2.364573</td>
      <td>1183.218216</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The mean difference in Bank Customer by Debt is : '+ str(BankCustomer_diff.loc[1,'Debt']-BankCustomer_diff.loc[0,'Debt']))
```

    The mean difference in Bank Customer by Debt is : 0.9811871805916113
    


```python
Gender_diff = credit.groupby('Gender')['Age','Debt','YearsEmployed','Income'].mean()
Gender_diff
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
      <th>YearsEmployed</th>
      <th>Income</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.886190</td>
      <td>5.072690</td>
      <td>1.785857</td>
      <td>1033.62381</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.788833</td>
      <td>4.621365</td>
      <td>2.414833</td>
      <td>1010.28125</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The mean difference in Gender by Debt is : '+ str(Gender_diff.loc[1,'Debt']-Gender_diff.loc[0,'Debt']))
```

    The mean difference in Gender by Debt is : -0.45132589285714353
    

As we can see the distributions for both ***Bank Customer*** and ***Gender*** variables, we have calculated the mean difference regarding some continues variables.The tables show there some differences in mean in respect to ***Debt*** and other variables, but we still don't know, are they statistically significant or not. 
To check this issue and answer the question above, we can consider the following hypothesis :

**Bank Customer and Debt :**

    - H0 : There is no differnece in Debt between being a bank customer or not
    - H1 : There is a significant difference in Debt between being a bank customer or not a bank customer
    
**Gender and Debt :**

    - H0 : There is no differnece in Debt between Male and Female
    - H1 : There is a significant difference in Debt between Male and Female

For this purposes we need to determine the p-value to check wether a variable is statistically significant or not. For getting the p-value I will shaffle the data or labels repeatedly and calculate the desired statistics. So, p-value can be in the range of 1% to 10% and that means we have confidence interval to accept or reject the null hypothesis.
Let's assume the range of $\alpha$ in ***(1%,5%,10%)***

# Bank Customer and Debt


```python
X = credit.BankCustomer
Y = credit.Debt
```


```python
Diff_Original_Customer = Yes_Customer.mean() - No_Customer.mean()
print('The difference in Balance by Gender (in the data) is: '+ str(Diff_Original_Customer))
```

    The difference in Balance by Gender (in the data) is: 0.9811871805916121
    


```python
dataframe= pd.DataFrame(X)
dataframe['Debt']=Y
dataframe.tail()
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
      <th>BankCustomer</th>
      <th>Debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>685</th>
      <td>0</td>
      <td>10.085</td>
    </tr>
    <tr>
      <th>686</th>
      <td>1</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>687</th>
      <td>0</td>
      <td>13.500</td>
    </tr>
    <tr>
      <th>688</th>
      <td>1</td>
      <td>0.205</td>
    </tr>
    <tr>
      <th>689</th>
      <td>1</td>
      <td>3.375</td>
    </tr>
  </tbody>
</table>
</div>




```python

def shuffling_Algo(data):
    cust = np.zeros(data.BankCustomer.count())
    cust[np.random.choice(data.BankCustomer.count(),int(sum(data.BankCustomer)),replace=False)] = 1
    data['BankCustomer'] = cust 
    return data

def mean_diff(data):
    return data.groupby('BankCustomer').mean().loc[0,'Debt'] - data.groupby('BankCustomer').mean().loc[1,'Debt']
```


```python
def sim_dist(frame, N=100):
    a = []
    for i in range(N):
        a.append(mean_diff(shuffling_Algo(dataframe)))
    return a

def plot_dist(dist,data,color='blue',bins=bins,orig=True):
    fig = plt.figure(figsize=(10,6))
    n, bins, patches = plt.hist(dist, bins = bins,  facecolor=color, alpha=0.5)
    values, base = np.histogram(dist, bins = bins)
    if orig:
        plt.axvline(np.mean(data), color=color, linestyle='dashed', linewidth=2,label='Original data')
        plt.axvline(-np.mean(data), color='red', linestyle='dashed', linewidth=2,label='Original data')
        plt.legend()
    plt.title('Mean difference')
```


```python
N = 1000
distribution = sim_dist(dataframe,N)
```


```python
plot_dist(distribution,Diff_Original_Customer,'green',100)
```


 <img src="/img/Statistical_Inference_2_24_0.png" alt="" />   

    



```python
# Calculating P-Value
def p_value(dist,estimation):
    return float(sum(np.array(dist)>estimation))/len(dist)
```


```python
pvalue = p_value(distribution,Diff_Original_Customer)
pvalue
```




    0.023



So, interesting. We have determined that the p-value and based on the selected $\alpha$ we can decide is the null hypothesis will be rejected or not. As the p_value is a bit more than 1%, with this percentage we cannot reject null hypothesis. Although choosing $\alpha$ depends on the context and the goal you deserve it, 1% for $\alpha$ is a strict choice. 
Considered other $\alpha$, ***5%*** which it is commonly used, we can reject null hypothesis and this means there is a significant difference in Debt between being a bank customer or not a bank customer that was obvious in the following figure [***bank customer has more debt that non-bank customer***](https://quantworld.netlify.app/img/Credit_Approval_Final_24_0.png).

Now let's repeat the experiment for Gender variable and see the result

# Gender and Debt 


```python
X = credit.Gender
Y = credit.Debt
```


```python
Diff_Original_Gender = Male.mean() - Female.mean()
print('The difference in Balance by Gender (in the data) is: '+ str(Diff_Original_Gender))
```

    The difference in Balance by Gender (in the data) is: -0.45132589285714264
    


```python
dataframe= pd.DataFrame(X)
dataframe['Debt']=Y
dataframe.tail()
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
      <th>Debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>685</th>
      <td>1</td>
      <td>10.085</td>
    </tr>
    <tr>
      <th>686</th>
      <td>0</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>687</th>
      <td>0</td>
      <td>13.500</td>
    </tr>
    <tr>
      <th>688</th>
      <td>1</td>
      <td>0.205</td>
    </tr>
    <tr>
      <th>689</th>
      <td>1</td>
      <td>3.375</td>
    </tr>
  </tbody>
</table>
</div>




```python
def shuffling_Algo(data):
    Gen = np.zeros(data.Gender.count())#.astype(float)
    Gen[np.random.choice(data.Gender.count(),int(sum(data.Gender)),replace=False)] = 1
    data['Gender'] = Gen 
    return data

def mean_diff(data):
    return data.groupby('Gender').mean().loc[0,'Debt'] - data.groupby('Gender').mean().loc[1,'Debt']
```


```python
N = 1000
distribution = sim_dist(dataframe,N)
```


```python
plot_dist(distribution,Diff_Original_Gender,'green',100)
```


 <img src="/img/Statistical_Inference_2_35_0.png" alt="" />       

    



```python
# Calculating P-Value
def p_value(dist,estimation):
    return float(sum(np.array(dist)>estimation))/len(dist)
```


```python
pvalue = p_value(distribution,Diff_Original_Gender)
pvalue
```




    0.861



The result for ***Gender*** variable shows that we cannot reject the null hypothesis with $\alpha$ = ***5%*** as p_value is very high. So, we can conclude that, there is no differnece in Debt between Male and Female
