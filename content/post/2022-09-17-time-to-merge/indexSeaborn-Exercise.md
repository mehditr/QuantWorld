---
title: Time to merge
author: ''
date: '2022-09-17'
slug: indexSeaborn-Exercise
categories: []
tags: []
subtitle: 'Excercises'
summary: ''
authors: []
lastmod: '2022-09-17T01:07:35+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
# Seaborn Exercises

## Imports

Run the cell below to import the libraries


```python
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
```

## The Data

DATA SOURCE: https://www.kaggle.com/rikdifos/credit-card-approval-prediction

Data Information:

Credit score cards are a common risk control method in the financial industry. It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank is able to decide whether to issue a credit card to the applicant. Credit scores can objectively quantify the magnitude of risk.

Feature Information:

<table>
<thead>
<tr>
<th>application_record.csv</th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Feature name</td>
<td>Explanation</td>
<td>Remarks</td>
</tr>
<tr>
<td><code>ID</code></td>
<td>Client number</td>
<td></td>
</tr>
<tr>
<td><code>CODE_GENDER</code></td>
<td>Gender</td>
<td></td>
</tr>
<tr>
<td><code>FLAG_OWN_CAR</code></td>
<td>Is there a car</td>
<td></td>
</tr>
<tr>
<td><code>FLAG_OWN_REALTY</code></td>
<td>Is there a property</td>
<td></td>
</tr>
<tr>
<td><code>CNT_CHILDREN</code></td>
<td>Number of children</td>
<td></td>
</tr>
<tr>
<td><code>AMT_INCOME_TOTAL</code></td>
<td>Annual income</td>
<td></td>
</tr>
<tr>
<td><code>NAME_INCOME_TYPE</code></td>
<td>Income category</td>
<td></td>
</tr>
<tr>
<td><code>NAME_EDUCATION_TYPE</code></td>
<td>Education level</td>
<td></td>
</tr>
<tr>
<td><code>NAME_FAMILY_STATUS</code></td>
<td>Marital status</td>
<td></td>
</tr>
<tr>
<td><code>NAME_HOUSING_TYPE</code></td>
<td>Way of living</td>
<td></td>
</tr>
<tr>
<td><code>DAYS_BIRTH</code></td>
<td>Birthday</td>
<td>Count backwards from current day (0), -1 means yesterday</td>
</tr>
<tr>
<td><code>DAYS_EMPLOYED</code></td>
<td>Start date  of employment</td>
<td>Count backwards from current day(0). If  positive, it means the person currently unemployed.</td>
</tr>
<tr>
<td><code>FLAG_MOBIL</code></td>
<td>Is there a mobile   phone</td>
<td></td>
</tr>
<tr>
<td><code>FLAG_WORK_PHONE</code></td>
<td>Is there a work phone</td>
<td></td>
</tr>
<tr>
<td><code>FLAG_PHONE</code></td>
<td>Is there a phone</td>
<td></td>
</tr>
<tr>
<td><code>FLAG_EMAIL</code></td>
<td>Is there an email</td>
<td></td>
</tr>
<tr>
<td><code>OCCUPATION_TYPE</code></td>
<td>Occupation</td>
<td></td>
</tr>
<tr>
<td><code>CNT_FAM_MEMBERS</code></td>
<td>Family size</td>
<td></td>
</tr>
</tbody>
</table>


```python
df = pd.read_csv('archive/application_record.csv')
```


```python
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
      <th>ID</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>DAYS_BIRTH</th>
      <th>DAYS_EMPLOYED</th>
      <th>FLAG_MOBIL</th>
      <th>FLAG_WORK_PHONE</th>
      <th>FLAG_PHONE</th>
      <th>FLAG_EMAIL</th>
      <th>OCCUPATION_TYPE</th>
      <th>CNT_FAM_MEMBERS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5008804</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>427500.0</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Civil marriage</td>
      <td>Rented apartment</td>
      <td>-12005</td>
      <td>-4542</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5008805</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>427500.0</td>
      <td>Working</td>
      <td>Higher education</td>
      <td>Civil marriage</td>
      <td>Rented apartment</td>
      <td>-12005</td>
      <td>-4542</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5008806</td>
      <td>M</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>112500.0</td>
      <td>Working</td>
      <td>Secondary / secondary special</td>
      <td>Married</td>
      <td>House / apartment</td>
      <td>-21474</td>
      <td>-1134</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Security staff</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5008808</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>270000.0</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>-19110</td>
      <td>-3051</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Sales staff</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5008809</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>270000.0</td>
      <td>Commercial associate</td>
      <td>Secondary / secondary special</td>
      <td>Single / not married</td>
      <td>House / apartment</td>
      <td>-19110</td>
      <td>-3051</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Sales staff</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 438557 entries, 0 to 438556
    Data columns (total 18 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   ID                   438557 non-null  int64  
     1   CODE_GENDER          438557 non-null  object 
     2   FLAG_OWN_CAR         438557 non-null  object 
     3   FLAG_OWN_REALTY      438557 non-null  object 
     4   CNT_CHILDREN         438557 non-null  int64  
     5   AMT_INCOME_TOTAL     438557 non-null  float64
     6   NAME_INCOME_TYPE     438557 non-null  object 
     7   NAME_EDUCATION_TYPE  438557 non-null  object 
     8   NAME_FAMILY_STATUS   438557 non-null  object 
     9   NAME_HOUSING_TYPE    438557 non-null  object 
     10  DAYS_BIRTH           438557 non-null  int64  
     11  DAYS_EMPLOYED        438557 non-null  int64  
     12  FLAG_MOBIL           438557 non-null  int64  
     13  FLAG_WORK_PHONE      438557 non-null  int64  
     14  FLAG_PHONE           438557 non-null  int64  
     15  FLAG_EMAIL           438557 non-null  int64  
     16  OCCUPATION_TYPE      304354 non-null  object 
     17  CNT_FAM_MEMBERS      438557 non-null  float64
    dtypes: float64(2), int64(8), object(8)
    memory usage: 60.2+ MB
    

## TASKS 

### Recreate the plots shown in the markdown image cells. Each plot also contains a brief description of what it is trying to convey. Note, these are meant to be quite challenging. Start by first replicating the most basic form of the plot, then attempt to adjust its styling and parameters to match the given image.

In general do not worry about coloring,styling, or sizing matching up exactly. Instead focus on the content of the plot itself. Our goal is not to test you on recognizing figsize=(10,8) , its to test your understanding of being able to see a requested plot, and reproducing it.

**NOTE: You may need to perform extra calculations on the pandas dataframe before calling seaborn to create the plot.**

----
----
### TASK: Recreate the Scatter Plot shown below

**The scatterplot attempts to show the relationship between the days employed versus the age of the person (DAYS_BIRTH) for people who were not unemployed. Note, to reproduce this chart you must remove unemployed people from the dataset first. Also note the sign of the axis, they are both transformed to be positive. Finally, feel free to adjust the *alpha* and *linewidth* parameters in the scatterplot since there are so many points stacked on top of each other.** 

#<img src="task_one.jpg">
<img src="C:/Users/Mehdi/Desktop/QuantWorld/static/Seaborn-Exercise_files"/>

```python
# CODE HERE TO RECREATE THE PLOT SHOWN ABOVE
df_just_employed = df.loc[df['DAYS_EMPLOYED'] < 0]
df_res = df_just_employed.loc[:,['DAYS_EMPLOYED','DAYS_BIRTH']].abs()

plt.figure(figsize=(10,4), dpi=100)
sns.scatterplot(x='DAYS_BIRTH',y='DAYS_EMPLOYED', data=df_res)
```




    <AxesSubplot:xlabel='DAYS_BIRTH', ylabel='DAYS_EMPLOYED'>




    
![png](./Seaborn-Excercise_20_1.png)
    



```python

```

-----
### TASK: Recreate the Distribution Plot shown below:

<img src="DistPlot_solution.png">

**Note, you will need to figure out how to calculate "Age in Years" from one of the columns in the DF. Think carefully about this. Don't worry too much if you are unable to replicate the styling exactly.**


```python
# CODE HERE TO RECREATE THE PLOT SHOWN ABOVE
df_year = abs(df.loc[:,'DAYS_BIRTH']) // 365

sns.displot(x=df_year, bins=50,color='#4A4A93')
plt.xlabel('Age in Years')
```




    Text(0.5, 6.79999999999999, 'Age in Years')




    
![png](./Seaborn-Exercise_13_1.png)
    



```python

```

-----
### TASK: Recreate the Categorical Plot shown below:

<img src='catplot_solution.png'>

**This plot shows information only for the *bottom half* of income earners in the data set. It shows the boxplots for each category of NAME_FAMILY_STATUS column for displaying their distribution of their total income. Note: You will need to adjust or only take part of the dataframe *before* recreating this plot. You may want to explore the *order* parameter to get the xticks in the exact order shown here**


```python
# CODE HERE
plt.figure(figsize=(10,4))
df_income = df.loc[df['AMT_INCOME_TOTAL'] < df['AMT_INCOME_TOTAL'].median()]
sns.boxplot(x='NAME_FAMILY_STATUS',y='AMT_INCOME_TOTAL', data=df_income, hue='FLAG_OWN_REALTY', palette='Set2')
plt.legend(loc=(1.03,0.5), title='FLAG_OWN_REALTY')
```




    <matplotlib.legend.Legend at 0x1b5d33efc10>




    
![png](./Seaborn-Exercise_16_1.png)
    


### TASK: Recreate the Heat Map shown below:

<img src='heatmap_solution.png'>

**This heatmap shows the correlation between the columns in the dataframe. You can get correlation with .corr() , also note that the FLAG_MOBIL column has NaN correlation with every other column, so you should drop it before calling .corr().**


```python
# CODE HERE
df_corr = df.drop('FLAG_MOBIL', axis=1)
sns.heatmap(df_corr.corr(),)
```




    <AxesSubplot:>




    
![png](./Seaborn-Exercise_19_1.png)
    



```python

```




    <AxesSubplot:>




    
![png](./Seaborn-Exercise_20_1.png)
    

