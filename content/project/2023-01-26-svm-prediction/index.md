---
title: SVM_Prediction
author: ''
date: '2023-01-26'
slug: Support Vector Machine - Stock Prediction
categories: [Machine Learning]
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2023-01-26T21:33:05+01:00'
featured: no
image:
  placement: 6
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

# Support Vector Machines Theory and Application:

In this report we will implement support vector machine to predict up trends for ETFs. Before going to the coding part I will discuss about support vector machine and what can we do with this ML technique.

Support vector machine is a popular supervised learning technique. It is a binary (or discriminative) classifier and it divides the data based on the position of predicted point in the classes relative to the hyperplane. SVM is based on the idea of finding a hyperplane that best separates the features into different areas. The best separates can be interpreted as maximum gap or maximum margins between two (or more) classes. In the two dimensional space, hyperplane is a line that divides the space in two parts A large functional margin would represent clear and confident classification.
The dividing hyperplane is all points such that:

 $$ \theta^{T} x + \theta_0\ = 0 $$ <br> 
where $\theta$ is the vector perpendicular to the hyperplane and $\theta_0 $ is a scalar.

Now suppose we have a function based on the definition above as $ g(\theta^{T} x + \theta_0\ )$. According to SVM and for $ z >0 $ $ g(z) =1 $ and for $ z < 0 $, $ g(z) = -1 $. So, we can demonstrate the binary classification as follows:
$ \theta^{T} x^{±} + \theta_0\ = ±1 $ or simplifying it as $ \theta^{T}(x^{+} - x^{-})$.

Let's assume $x^{n}$ as M-dimensional vector representing $ n^{th} $ samples and $ y^{n} $ is the classified labels as +1 and -1. After multiplying the function of 1 and -1 we get the following expression:
 $$ y^{n} ( \theta^{T} x^{n} + \theta_0 ) -1 \geq  0 $$
That is also known as :        $\theta^{T} (x^{+} - x^{-}) = 2$

The last expression follows that : 
$$ Margin Width = \frac{2}{|\theta|}$$

##### Hard Margin <br>
If the data is linearly separable, two parallel hyperplanes can be selected and separate the two labels of data, so that the distance between them is as large as possible. <br>
So, for maximising the margins, we have to minimise find $\theta$ and $\theta_0$ that minimises $|\theta|$ :
$$  \min\limits_{\theta, \theta_0} \frac{1}{2}|\theta|^{2} $$

Subject to  :   
$$   y_i (\theta^T x_i + \theta_0 ) \geq 1 $$

##### Soft Margin <br>
When the data is not linearly separable, we need somehow allow the algorithm to cope with this problem. Therefore, we introduce $\xi_i$ term and put it both in the optimization funtion and constraint. In our optimization function, we also see hyperparameter $C$ to account for $\xi_i$ term. After substituting $\xi_i$ in our optimiation problem, we obtain unconditional optimization. Thus, we obtain hidge loss function and L2 regularization.
$$ \frac{1}{2}|\theta|^{2} + C \sum_{i=1}^l \xi_i   \longrightarrow \min\limits_{\theta,\theta_0, \xi} $$ <br>
subject to:
$$ y_i (\theta^T x_i + \theta_0 ) \geq 1 - \xi_i $$
$$ \xi_i \geq 0$$

In support vector machine we have several hyper parameters such as C which is hyperplane and we talked about that, kernel which we can determine the relation between train and test data and consists of four items 1-linear, 2-Radial (rbf), 3- poly and 4- sigmoid. The last one is gamma which very important and with low gamma, points far away from plausible separation line are considered in calculation for the separation line. Whereas high gamma means the points close to plausible line are considered in calculation.



# 1- Understand the problem:
In this project we are going to be predicting the uptrend movement of a ticker by support vector machines. Since this technique requires the target to be classified into two classes as 1 and 0 however during the project, we sometimes consider the total movements in comparison with the uptrend direction. But our goal is to consider the last result just for uptrends. 

We will start first by loading the modules needed



```python
# Data manipulation 
import pandas as pd
import numpy as np

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')

# Import Data
import yfinance as yf 

# Preprocessing & Cross validation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#Classifier
from sklearn.svm import SVC 
# Metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, plot_roc_curve,confusion_matrix,roc_curve
from sklearn.metrics import recall_score, precision_score, f1_score
# Feature Selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif

import pyfolio as pf
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
```

# 2- Collect data:

I have selected, The Invesco QQQ ETF for this project, which tracks the Nasdaq-100 Index, ranks in the top 1% of large-cap growth-funds. The index includes the 100 largest non-financial companies listed on the Nasdaq, based on the market cap. Since its formation in 1999, QQQ has demonstrated a successful path and typically beating the S&P500 Index.


```python
QQQ_data = yf.download("QQQ", start="2014-01-01", end="2020-02-28")
QQQ=QQQ_data['Adj Close']
```

    [*********************100%***********************]  1 of 1 completed
    

# 3-Visualization, Creating Features and Functions:

For the first step, we have look at the graph in the determined date:


```python
QQQ_data['Adj Close'].plot()
```




    <AxesSubplot:xlabel='Date'>




<img src="/img/SVM_Prediction_10_1.png" alt="" />      

    


So, there is huge fall between 2017 and 2018 and then the price started going up and then went down again.
Now, we are going to build the auxiliary functions for creating the features. 
The fist function computes log returns in different lags. We will make 5 lags of the length of one day. We called ‘ret_0’ as the most recent days. In the function below, we see an argument ‘length_win’ that can be used for even 1 week or more returns, however we use daily price to get 5 daily returns.



```python
def LagRet(price,lag=5,lenght_win = 1):
    
    lagRet = pd.DataFrame(index=price.index)
    lagRet['Adj Close']= price
    lagRet['ret_0']= np.log(lagRet['Adj Close']/lagRet['Adj Close'].shift(lenght_win))
    
    
    for lag in range(1, lag+1):
        col = 'ret_%d'%lag
        lagRet[col] = lagRet['ret_0'].shift(lag)
    lagRet.dropna(inplace=True)
    
    return lagRet
```

Then we will get the first set of our features consist of daily returns


```python
QQQ_features = LagRet(QQQ)
QQQ_features.head()
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
      <th>Adj Close</th>
      <th>ret_0</th>
      <th>ret_1</th>
      <th>ret_2</th>
      <th>ret_3</th>
      <th>ret_4</th>
      <th>ret_5</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-10 00:00:00-05:00</th>
      <td>80.790894</td>
      <td>0.003212</td>
      <td>-0.003327</td>
      <td>0.002178</td>
      <td>0.009225</td>
      <td>-0.003700</td>
      <td>-0.007245</td>
    </tr>
    <tr>
      <th>2014-01-13 00:00:00-05:00</th>
      <td>79.597107</td>
      <td>-0.014887</td>
      <td>0.003212</td>
      <td>-0.003327</td>
      <td>0.002178</td>
      <td>0.009225</td>
      <td>-0.003700</td>
    </tr>
    <tr>
      <th>2014-01-14 00:00:00-05:00</th>
      <td>81.114799</td>
      <td>0.018888</td>
      <td>-0.014887</td>
      <td>0.003212</td>
      <td>-0.003327</td>
      <td>0.002178</td>
      <td>0.009225</td>
    </tr>
    <tr>
      <th>2014-01-15 00:00:00-05:00</th>
      <td>81.781136</td>
      <td>0.008181</td>
      <td>0.018888</td>
      <td>-0.014887</td>
      <td>0.003212</td>
      <td>-0.003327</td>
      <td>0.002178</td>
    </tr>
    <tr>
      <th>2014-01-16 00:00:00-05:00</th>
      <td>81.790375</td>
      <td>0.000113</td>
      <td>0.008181</td>
      <td>0.018888</td>
      <td>-0.014887</td>
      <td>0.003212</td>
      <td>-0.003327</td>
    </tr>
  </tbody>
</table>
</div>



Let’s examine the ret_0 in the qq-plot. It shows that the most recent stock return has fat tails. This is almost true for other lagged returns. This can be useful when a scaler will be chosen.


```python
sm.qqplot(QQQ_features['ret_0'],fit= True, line='45')
plt.title("QQQ",fontweight="bold")
```




    Text(0.5, 1.0, 'QQQ')




<img src="/img/SVM_Prediction_16_1.png" alt="" />    
    


The next set of features we use in momentum. So we compute and add momentum indicators for 5-days, 7-days, 13-days, and 21-days which frequently used by traders.


```python
def GetMomentum(tar_df,s_df,time_int=[5,7,13,21]):
    for time_interval in time_int:
        momentum = (s_df.shift(time_interval)['Adj Close']-s_df['Adj Close'])/s_df['Adj Close']
        tar_df = pd.DataFrame(tar_df.iloc[time_interval:,:])
        tar_df['MOM'+str(time_interval)]=momentum
    
    tar_df = tar_df.dropna()
    return tar_df
```


```python
QQQ_features = GetMomentum(QQQ_features,QQQ_features)
QQQ_features.head()
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
      <th>Adj Close</th>
      <th>ret_0</th>
      <th>ret_1</th>
      <th>ret_2</th>
      <th>ret_3</th>
      <th>ret_4</th>
      <th>ret_5</th>
      <th>MOM5</th>
      <th>MOM7</th>
      <th>MOM13</th>
      <th>MOM21</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-03-19 00:00:00-04:00</th>
      <td>83.672501</td>
      <td>-0.005427</td>
      <td>0.012002</td>
      <td>0.008758</td>
      <td>-0.006856</td>
      <td>-0.014235</td>
      <td>0.003761</td>
      <td>0.005775</td>
      <td>0.006331</td>
      <td>0.003332</td>
      <td>-0.001812</td>
    </tr>
    <tr>
      <th>2014-03-20 00:00:00-04:00</th>
      <td>83.904800</td>
      <td>0.002772</td>
      <td>-0.005427</td>
      <td>0.012002</td>
      <td>0.008758</td>
      <td>-0.006856</td>
      <td>-0.014235</td>
      <td>-0.011186</td>
      <td>-0.000775</td>
      <td>-0.006867</td>
      <td>-0.011413</td>
    </tr>
    <tr>
      <th>2014-03-21 00:00:00-04:00</th>
      <td>82.895195</td>
      <td>-0.012106</td>
      <td>0.002772</td>
      <td>-0.005427</td>
      <td>0.012002</td>
      <td>0.008758</td>
      <td>-0.006856</td>
      <td>-0.005981</td>
      <td>0.015206</td>
      <td>0.018009</td>
      <td>0.005316</td>
    </tr>
    <tr>
      <th>2014-03-24 00:00:00-04:00</th>
      <td>82.168686</td>
      <td>-0.008803</td>
      <td>-0.012106</td>
      <td>0.002772</td>
      <td>-0.005427</td>
      <td>0.012002</td>
      <td>0.008758</td>
      <td>0.011629</td>
      <td>0.009706</td>
      <td>0.029837</td>
      <td>0.012853</td>
    </tr>
    <tr>
      <th>2014-03-25 00:00:00-04:00</th>
      <td>82.438789</td>
      <td>0.003282</td>
      <td>-0.008803</td>
      <td>-0.012106</td>
      <td>0.002772</td>
      <td>-0.005427</td>
      <td>0.012002</td>
      <td>0.020489</td>
      <td>-0.000478</td>
      <td>0.025561</td>
      <td>0.014923</td>
    </tr>
  </tbody>
</table>
</div>



Finally we calculate the set of moving averages in different formats. We will add simple and exponential moving averages and we will also add crossover strategies, short terms against long terms, for all exponential and simple moving averages.  The last feature is 21-days rolling standard deviation of returns.


```python
#Compute EWMA for 5,7,13,21 days and make crossover strategies
QQQ_features['EWMA5'] = pd.Series(QQQ_features['Adj Close']).ewm(span = 5).mean()
QQQ_features['EWMA7'] = pd.Series(QQQ_features['Adj Close']).ewm(span = 7).mean()
QQQ_features['EWMA13'] = pd.Series(QQQ_features['Adj Close']).ewm(span = 13).mean()
QQQ_features['EWMA21'] = pd.Series(QQQ_features['Adj Close']).ewm(span = 21).mean()
QQQ_features['EWMA5-7'] = QQQ_features['EWMA5'] - QQQ_features['EWMA7']
QQQ_features['EWMA7-13'] = QQQ_features['EWMA7'] - QQQ_features['EWMA13']
QQQ_features['EWMA13-21'] = QQQ_features['EWMA13'] - QQQ_features['EWMA21']

#Compute Simple MA for 5,7,13,21 days and make crossover strategies
QQQ_features['MA5'] = pd.Series(QQQ_features['Adj Close']).rolling(5).mean()
QQQ_features['MA7'] = pd.Series(QQQ_features['Adj Close']).rolling(7).mean()
QQQ_features['MA13'] = pd.Series(QQQ_features['Adj Close']).rolling(13).mean()
QQQ_features['MA21'] = pd.Series(QQQ_features['Adj Close']).rolling(21).mean()
QQQ_features['MA5-7'] = QQQ_features['MA5'] - QQQ_features['MA7']
QQQ_features['MA7-13'] = QQQ_features['MA7'] - QQQ_features['MA13']
QQQ_features['MA13-21'] = QQQ_features['MA13'] - QQQ_features['MA21']

#standard deviations on 21-days rolling windows
QQQ_features['stdev21'] = pd.Series(QQQ_features['Adj Close']).rolling(21).std()




```

Now, I will add the target variable which is classified as 1 for uptrend movements and 0 for constant or downtrend movements. After that I will drop all ‘NaNs’ generated by moving averages and rolling windows and then check the column names


```python
# set Sign for uptrend movements
QQQ_features['Sign'] = QQQ_features['ret_0'].apply(lambda x: 1.0 if x >0 else 0.0)


print(QQQ_features.columns)
QQQ_features= QQQ_features.dropna()
```

    Index(['Adj Close', 'ret_0', 'ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5',
           'MOM5', 'MOM7', 'MOM13', 'MOM21', 'EWMA5', 'EWMA7', 'EWMA13', 'EWMA21',
           'EWMA5-7', 'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21',
           'MA5-7', 'MA7-13', 'MA13-21', 'stdev21', 'Sign'],
          dtype='object')
    

After completing the dataset, it is worth to check the correlation between the features. But we know that as we have some homogeneous features such as MA and EWMA they will have a high correlation together. This also can have bad effect on the model.


```python
corrmat = QQQ_features.drop(['Adj Close','ret_0','Sign'],axis=1).corr()
fig, ax = plt.subplots(figsize=(16,6))

mask = np.triu(np.ones_like(corrmat,dtype=bool))

cmap = sns.diverging_palette(250,15,as_cmap=True)

sns.heatmap(corrmat,annot=True,annot_kws={'size':10},
           fmt='0.2f',mask=mask,cmap=cmap,vmax=0.3,center=0,
           square=False,linewidths=0.5,cbar_kws={'shrink':1})

ax.set_title('Feature Correlation', fontsize=14,color='black')
```




    Text(0.5, 1.0, 'Feature Correlation')




<img src="/img/SVM_Prediction_25_1.png" alt="" />    
    


As we expected, there are high correlation relations between MA and EWMA and it is known by red colors.

So before going to the transform section, and in order to have a better understanding of the relationship between the variables and the output of the model, first we apply the model on a number of features and consider them numerically and graphically. This is good for having the first impression of the changes we will make in the model in order to get the best and pragmatic solution.

So, I will create a train and validation datasets but now, I am not going to scale them as I am just going to have the first impression

I will use 75% of total dataset and use it for my purposes.


```python
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]

QQQ_validate = QQQ_features.iloc[training_split:,:]
```

We will start the prediction with the first 5 lagged returns and testing C with 1000 and 10 respectively. The first obviously might be a high number for C and we check the shape and the accuracy that will be changed. Generally when C increases, because we penalize the model more, then overfitting increases, and if C decreases then underfitting increases. Basically, if C has a large value the optimizer is trying to find smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, for the small value of C, the optimizer is looking for a larger margin separating hyperplane, even if that hyperplane misclassifies more points.


**The whole movements : Both positive and negative trends for C = 1000 and C = 10** 


```python

cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5']
SVM_QQQ = SVC(C=1000,probability=True)

#fit the model

SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

# Predicting with validation set
    
QQQ_validate['SVM_Predict']=SVM_QQQ.predict(QQQ_validate[cols])

QQQ_validate['SVM_Predict'][QQQ_validate['SVM_Predict']==0]=-1
QQQ_validate['SVM_Returns']= QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

crossval_QQQ =cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring ='accuracy')
print("Accuracy For QQQ_SVM Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(16,6))
plt.title('QQQ - Positive and Negative trends with C = 1000',fontweight='bold')

plt.show()
```

    Accuracy For QQQ_SVM Model : 0.5436
    


<img src="/img/SVM_Prediction_33_1.png" alt="" />    

    



```python
# change number C as hyperplane to check the changes in the fjgure
# C=10
cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5']
SVM_QQQ = SVC(C=10, probability=True)

#fit the model

SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

# Predicting with validation set

QQQ_validate['SVM_Predict']=SVM_QQQ.predict(QQQ_validate[cols])

QQQ_validate['SVM_Predict'][QQQ_validate['SVM_Predict']==0]=-1
QQQ_validate['SVM_Returns']= QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

crossval_QQQ =cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring ='accuracy')
print("Accuracy For QQQ_SVM Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(16,6))
plt.title('QQQ - Positive and Negative trends with C = 10',fontweight='bold')

plt.show()
```

    Accuracy For QQQ_SVM Model : 0.5586
    


<img src="/img/SVM_Prediction_34_1.png" alt="" />    

    


**The Uptrend movements : Both positive trends for C = 1000 and C = 10**


```python

cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5']
SVM_QQQ = SVC(C=1000, probability=True)

#fit the model

SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

# Predicting with validation set
    
QQQ_validate['SVM_Predict']=SVM_QQQ.predict(QQQ_validate[cols])


QQQ_validate['SVM_Returns']= QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']
QQQ_validate.tail(3)

crossval_QQQ =cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring ='accuracy')
print("Accuracy For QQQ_SVM Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(16,6))
plt.title('QQQ - Positive trends with C = 1000 ',fontweight='bold')

plt.show()
```

    Accuracy For QQQ_SVM Model : 0.5436
    


<img src="/img/SVM_Prediction_36_1.png" alt="" />    

    



```python
# change number C as hyperplane to check the changes in the fjgure
# C=10
cols=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5']
SVM_QQQ = SVC(C=10, probability=True)

#fit the model

SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

# Predicting with validation set

QQQ_validate['SVM_Predict']=SVM_QQQ.predict(QQQ_validate[cols])


QQQ_validate['SVM_Returns']= QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']
QQQ_validate.tail(3)

crossval_QQQ =cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring ='accuracy')
print("Accuracy For QQQ_SVM Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns']].cumsum().apply(np.exp).plot(figsize=(16,6))
plt.title('QQQ - Positive trends with C = 10 ',fontweight='bold')

plt.show()
```

    Accuracy For QQQ_SVM Model : 0.5586
    


<img src="/img/SVM_Prediction_37_1.png" alt="" />        

    


This model, as a first attempt, is not too bad. We have tried to show the result with different Cs with 1000 and 10 and in both trend directions and just positive trend movements. As we can see if C  increases  as I mentioned in the previous paragraph, the smaller margin is selected and fewer points are in the calculation in comparison with smaller C. Then C with smaller value is known as soft margin is less penalized and uderfitted but follow the original return more.

The notable point here is the accuracy of the model with C = 10 is better than C = 1000. This is not we expected to see as we penalize the model, we are going to get the higher accuracy. This can have several reasons such as the selected features, or the kernels and etc. But the aim now is to see how the shapes with different Cs behave.


The notable point here is the accuracy of the model with C = 10 is better than C = 1000. This is not we expected to see as we penalize the model, we are going to get the higher accuracy. This can have several reasons such as the selected features, or the kernels and etc. But the aim now is to see how the shapes with different Cs behave.


```python

cols1 = [ 'ret_1']
cols2 = [ 'MOM5' ]
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]

QQQ_validate = QQQ_features.iloc[training_split:,:]

SVM_QQQ =SVC(C=1000,probability=True, kernel="linear")
#fit again
SVM_QQQ.fit(QQQ_training[cols1],QQQ_training['Sign'])

SVM_QQQ.fit(QQQ_training[cols2],QQQ_training['Sign'])

#predict using the validation set of data
QQQ_validate['SVM_Pred_Ret1'] = SVM_QQQ.predict(QQQ_validate[cols1])

QQQ_validate['SVM_Pred_Mom5'] = SVM_QQQ.predict(QQQ_validate[cols2])


QQQ_validate['SVM_Returns_Ret1'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Pred_Ret1']

QQQ_validate['SVM_Returns_Mom5'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Pred_Mom5']


crossval_QQQ = cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring = 'accuracy')
print("Accuracy For QQQ SVM Hard Margins Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns_Mom5', 'SVM_Returns_Ret1']].cumsum().apply(np.exp).plot(figsize=(16, 6));
plt.title("QQQ Positive trends with C = 1000",fontweight="bold")
plt.show()
```

    Accuracy For QQQ SVM Hard Margins Model : 0.5680
    


<img src="/img/SVM_Prediction_40_1.png" alt="" />            

    


Although the accuracy is somehow good, but we see that momentum outperforms all others. Note that we can see slightly good prediction between predicted return and ‘ ret_0’ and sometimes they cross each other in some parts but they slightly move together.


```python
# SVM comparing Ret1 and Momentum 
# C=10
cols1 = [ 'ret_1']
cols2 = [ 'MOM5' ]
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]

QQQ_validate = QQQ_features.iloc[training_split:,:]

SVM_QQQ =SVC(C=10,probability=True, kernel="linear")
#fit again
SVM_QQQ.fit(QQQ_training[cols1],QQQ_training['Sign'])

SVM_QQQ.fit(QQQ_training[cols2],QQQ_training['Sign'])

#predict using the validation set of data
QQQ_validate['SVM_Pred_Ret1'] = SVM_QQQ.predict(QQQ_validate[cols1])

QQQ_validate['SVM_Pred_Mom5'] = SVM_QQQ.predict(QQQ_validate[cols2])


QQQ_validate['SVM_Returns_Ret1'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Pred_Ret1']

QQQ_validate['SVM_Returns_Mom5'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Pred_Mom5']


crossval_QQQ = cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring = 'accuracy')
print("Accuracy For QQQ SVM Soft Margins Model : %.4f"%crossval_QQQ.mean())

QQQ_validate[['ret_0','SVM_Returns_Mom5', 'SVM_Returns_Ret1']].cumsum().apply(np.exp).plot(figsize=(16, 6));
plt.title("QQQ Positive trends with C = 10",fontweight="bold")
plt.show()
```

    Accuracy For QQQ SVM Soft Margins Model : 0.5680
    


 <img src="/img/SVM_Prediction_42_1.png" alt="" />   

    


The accuracy is the same as previous model but we can see an improvement slopes in ret_0 and returns predicted by SVM. 
In general, we have seen that, the accuracy with linear kernel is somehow at least logical and slightly better. But this is just a change in variables and kernel. 

Since we would like to see the impact of two or more variables on the final result, we will make a prediction based on the combined previous variables and inspect the model. Further, we will see the impact of different between Cs and different kernels on the graph. We will use 2D visualization to check the impact of changes.


```python
def plot_svm(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
```


```python
# graph for SVM 

cols = [ 'ret_1', 'MOM5' ]
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]

QQQ_validate = QQQ_features.iloc[training_split:,:]

SVM_QQQ =SVC(C=1000,probability=True,kernel='linear')
#fit again
SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

#predict using the validation set of data
QQQ_validate['SVM_Predict'] = SVM_QQQ.predict(QQQ_validate[cols])


QQQ_validate['SVM_Returns'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

QQQ_validate.tail(3)

crossval_QQQ = cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring = 'accuracy')
print("Accuracy For QQQ SVM Hard Margins Model : %.4f"%crossval_QQQ.mean())

#print the support vectors
print(' Support Vectors for QQQ Standard Deviation')
df = pd.DataFrame(SVM_QQQ.support_vectors_)
df.columns = cols
print(df.std())
```

    Accuracy For QQQ SVM Hard Margins Model : 0.6587
     Support Vectors for QQQ Standard Deviation
    ret_1    0.009547
    MOM5     0.015527
    dtype: float64
    


```python

plt.figure(figsize=(16,6))  
plt.scatter(QQQ_validate['ret_1'], QQQ_validate['MOM5'], c=QQQ_validate['Sign'], s=len(QQQ_validate["ret_1"]), cmap='autumn')
plt.title('')
plot_svm(SVM_QQQ);

```


 <img src="/img/SVM_Prediction_47_0.png" alt="" />       

    


As we can see there is no clear separation between the two groups of data observations, some points appear to be on the wrong side of the hyperplane. It is obvious some points that can be determined as 'misclassified points' are not too far from the hyperplane. By increasing the margins we might improve the model but not in the accuracy however visually. Increasing the margins means we assign smaller number to C and basically the accuracy will decrease. So, we will check it in the next code.


```python
# graph for SVM 
cols = [ 'ret_1', 'MOM5' ]
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]

QQQ_validate = QQQ_features.iloc[training_split:,:]

SVM_QQQ =SVC(C=10,probability=True,kernel='linear')
#fit again
SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])

#predict using the validation set of data
QQQ_validate['SVM_Predict'] = SVM_QQQ.predict(QQQ_validate[cols])

QQQ_validate['SVM_Returns'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

QQQ_validate.tail(3)

crossval_QQQ = cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring = 'accuracy')
print("Accuracy For QQQ SVM Soft Margins Model : %.4f"%crossval_QQQ.mean())

#print the support vectors
print(' Support Vectors for QQQ Standard Deviation')
df = pd.DataFrame(SVM_QQQ.support_vectors_)
df.columns = cols
df.std()
```

    Accuracy For QQQ SVM Soft Margins Model : 0.6168
     Support Vectors for QQQ Standard Deviation
    




    ret_1    0.009774
    MOM5     0.017335
    dtype: float64




```python

plt.figure(figsize=(16,6))  
plt.scatter(QQQ_validate['ret_1'], QQQ_validate['MOM5'], c=QQQ_validate['Sign'], s=len(QQQ_validate["ret_1"]), cmap='autumn')
plot_svm(SVM_QQQ);

```


<img src="/img/SVM_Prediction_50_0.png" alt="" />          

    


In the comparison between the two figures and numbers above, the standard deviation has slightly increased in both 'ret_1' and 'MOM5', in the second figure; however when C, the classifier accepts more misclassified points and is okay with them because with small C we have high bias and low variance. This did not happen here and we have more variance with smaller C. We can see that there are some clear outliers. As it is obvious in the graph, by increasing margins we cover more points in the model however some of them are outliers. We should note that, the higher value of the C, the more SVM model will be sensitive to noise and hence.

In conclusion, C with larger numbers has narrower margins and many points can be considered as outliers, however for messy data, when we increase C as a regularization parameter, we deserve more accuracy. When C has smaller number more points come to the model for evaluation and we penalize the model less. In the dataset we are working on, there is no clear separation and we have focused on C to penalize. However as the dataset is not linearly separable, it is better other kernels such as 'rbf' be examined in the model.



Now, we apply 'non-linear' kernel and more specific 'rbf' kernel and see how the plots will be changed.

**Plot 2D Plot**


```python
# graph for SVM and non-linear kernel
# C= 1000
cols = ['ret_1','MOM5']
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]
QQQ_validate = QQQ_features.iloc[training_split:,:]
SVM_QQQ =SVC(C=1000,probability=True,kernel='rbf')
#fit again
SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])
#predict using the validation set of data
QQQ_validate['SVM_Predict'] = SVM_QQQ.predict(QQQ_validate[cols])
QQQ_validate['SVM_Predict'][QQQ_validate['SVM_Predict']==0] = -1
QQQ_validate['SVM_Returns'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

crossval_QQQ = cross_val_score(SVM_QQQ,QQQ_features[cols],QQQ_features['Sign'],scoring = 'accuracy')
print("Accuracy For QQQ SVM Hard Margins Model : %.4f"%crossval_QQQ.mean())

plt.figure(figsize=(15,6))
plt.scatter(QQQ_validate['ret_1'], QQQ_validate['MOM5']
,c=QQQ_validate['Sign'], s=len(QQQ_validate["ret_1"]), cmap='autumn')
plot_svm(SVM_QQQ);
```

    Accuracy For QQQ SVM Hard Margins Model : 0.6533
    


<img src="/img/SVM_Prediction_54_1.png" alt="" />              

    


The accuracy of ‘rbf’ kernel is close to the linear with the same number of C for the analysis. Although, the plot here might not a complete sense of the variable, it gives better sense relative to linear kernel. Let’s see the graph for soft margins and then will discuss about them again.


```python
# graph for SVM  and non-linear kernel
# C= 10
cols = ['ret_1','MOM5']
training_split = int(0.75*QQQ.shape[0])
QQQ_training = QQQ_features.iloc[0:training_split,:]
QQQ_validate = QQQ_features.iloc[training_split:,:]
SVM_QQQ =SVC(C=10,probability=True,kernel='rbf')
#fit again
SVM_QQQ.fit(QQQ_training[cols],QQQ_training['Sign'])
#predict using the validation set of data
QQQ_validate['SVM_Predict'] = SVM_QQQ.predict(QQQ_validate[cols])
QQQ_validate['SVM_Predict'][QQQ_validate['SVM_Predict']==0] = -1
QQQ_validate['SVM_Returns'] =QQQ_validate['ret_0']*QQQ_validate['SVM_Predict']

plt.figure(figsize=(15,6))
plt.scatter(QQQ_validate['ret_1'], QQQ_validate['MOM5']
,c=QQQ_validate['Sign'], s=len(QQQ_validate["ret_1"]), cmap='autumn')
plot_svm(SVM_QQQ);
```


<img src="/img/SVM_Prediction_56_0.png" alt="" />                  

    


It seems non-linear kernel ('rbf' here) is more flexible, however there is the risk of overfitting especially when we penalize the model more (higher number of C). From the first plot, we have penalized the model more and used 'rbf' kernel, the accuracy is close to the linear one. We can see, in the both plots that model tries to capture every single point. That can increase the complexity of the model and also as I mentioned before, that might be the sign of the overfitting.

So far we have seen how the regularization parameter known as C and kernels change the accuracy of model based on some of primary variables. Although, there are other hyperparamters such gamma can be tuned, other measures such as recall, precision and f1_score that sometimes are more important than accuracy, and other variables that are less correlated have not been seen yet. 

#  4&5:  Transformation and Feature engineering:

We will put all variables except 'ret_0' and 'Sign' column and after that; we will split the dataset into train and test. Before we have been working on training and validating datasets, but now we use the last 25% as test size of dataset. As the data in returns and moving averages and momentums are different in respect of numbers, we will have transform it. For transformation, I will use MinMaxScaler method instead of StandardScaler as it is illustrated before in qq plot the data is not normal and follows fat tails distributions.


```python
cols_full=['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5', 'EWMA13', 'EWMA21', 'EWMA5-7','MOM5', 'MOM7','MOM13', 'MOM21', 'EWMA5', 'EWMA7',
'EWMA7-13', 'EWMA13-21', 'MA5', 'MA7', 'MA13', 'MA21', 'MA5-7','MA7-13', 'MA13-21', 'stdev21']
```


```python
X_train_QQQ,X_test_QQQ,y_train_QQQ,y_test_QQQ = train_test_split(QQQ_features[cols_full], QQQ_features['Sign'],
                                                                 test_size=0.25, shuffle=False)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_X_scaled = scaler.fit_transform(X_train_QQQ)
test_X_scaled = scaler.transform(X_test_QQQ)

train_X_scaled = pd.DataFrame(train_X_scaled,columns=X_train_QQQ.columns)
test_X_scaled = pd.DataFrame(test_X_scaled,columns=X_train_QQQ.columns)

```

I will just examine, is the data in train or test imbalanced or not?


```python
y_train_QQQ.value_counts()
```




    1.0    625
    0.0    482
    Name: Sign, dtype: int64




```python
y_test_QQQ.value_counts()
```




    1.0    214
    0.0    156
    Name: Sign, dtype: int64



The data are not imbalanced in both train test data sets. In the train and test sets the division are around both 57% and 44% respectively

Before we go to the modeling part, I will test some features and see which features might be good for the final model. This is clear that we will run the model for both full variables and selected variables in modeling part and compare the results.  

As linear kernel of SVM is very similar to logistic regression, the effect of multicollinearity has a similar effect in linear kernel in SVM. So, first I will process the univariate feature selection in linear form and they examine the accuracy of the model without eliminating and with eliminating the variables. 


```python
X_indices = np.arange(1,25,1)
selector = SelectKBest(f_classif, k=5)
selector.fit(train_X_scaled, y_train_QQQ)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
```


```python
plt.bar( X_indices,scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='green',
        edgecolor='black')
plt.title('Univariate Feature Selection for SVM - linear')
plt.xlabel('Feature Number')
plt.xlabel('Univariate Score')
```




    Text(0.5, 0, 'Univariate Score')




<img src="/img/SVM_Prediction_69_1.png" alt="" />    

    


Based on the univariate feature selection process, the most important features are 8, 9 10, 11 and 12 and the number of 15 and 20 are slightly relative to others. These features are MOM5, MOM7, MOM13, MOM21 and EWMA5.

Now I will test the accuracy of model with all features and see what will be the accuracy for the model without eliminating the features.


```python

clf = LinearSVC(C=10, max_iter=10000)
clf.fit(train_X_scaled, y_train_QQQ)
print('Classification accuracy without selecting features: {:.3f}'
      .format(clf.score(test_X_scaled, y_test_QQQ)))

svm_weights = np.abs(clf.coef_[-1]).sum(axis=0)
svm_weights /= svm_weights.sum()

plt.bar(X_indices - .5, svm_weights, width=.5, label='SVM weight',
        color='green', edgecolor='blue')

plt.title('All features and without any optimization')
plt.xlabel('Feature Number')
plt.ylabel('Weight')
```

    Classification accuracy without selecting features: 0.908
    




    Text(0, 0.5, 'Weight')




<img src="/img/SVM_Prediction_72_2.png" alt="" />        

    


Now let’s try what will be the accuracy and features after features elimination.


```python
from sklearn.pipeline import make_pipeline

clf_selected = make_pipeline(SelectKBest(f_classif, k=5), LinearSVC(C=10, max_iter= 10000))
clf_selected.fit(train_X_scaled, y_train_QQQ)
print('Classification accuracy after univariate feature selection: {:.3f}'
      .format(clf_selected.score(test_X_scaled, y_test_QQQ)))

svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='c',
        edgecolor='black')

plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.ylabel('Weight')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()

idx_selected_features = X_indices[selector.get_support()].tolist()
svl_linear_best = clf_selected
```

    Classification accuracy after univariate feature selection: 0.568
    


<img src="/img/SVM_Prediction_74_1.png" alt="" />           

    


We have tried to find optimal features just by using univariate feature selection with F-test for feature scoring and plotted the result. In the first glimpse, almost all features somehow have been selected but some of them are more significant than others. Then we use the accuracy of the model by 'LinearSVC' before and after feature selection, but it seems the accuracy is much better than elimination. The accuracy without elimination is around 91% and the accuracy with elimination is around 56%.
Here there is a critical dilemma that should we continue without any selection as the accuracy is high or we have to think about this and get a decision? This question can be answered when we apply our models and face the results.

There are of course several methods to consider. One can use PCA (dimensionality reduction) instead of feature selection, although they are not the same. One can use logistic regression for selecting the features, but here I have applied the linear approach with SVC and just 5 selections. The reason I have chosen 5 selections (k=5) is the model will be interpretable.


# **6 & 7 - The Model Evaluation, Deploy and Communication for all variables:**

In the evaluation method, we will run the model first all by defaults on training data and then we will apply it on test data. Then we will try to find the best parameters with 'GridSearchCV' and then rerun the model and extract the metric measures.


```python
SVM_QQQ =SVC(probability=True)
SVM_QQQ.fit(train_X_scaled,y_train_QQQ)

```




    SVC(probability=True)




```python
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
y_pred_train = SVM_QQQ.predict(train_X_scaled)
print('Train Accuracy Score : ' + str(accuracy_score(y_train_QQQ,y_pred_train)))
print('Train Precision Score : ' + str(precision_score(y_train_QQQ,y_pred_train)))
print('Train Recall Score : ' + str(recall_score(y_train_QQQ,y_pred_train)))
print('Train F1 Score : ' + str(f1_score(y_train_QQQ,y_pred_train)))
print(classification_report(y_train_QQQ,y_pred_train))
```

    Train Accuracy Score : 0.8102981029810298
    Train Precision Score : 0.765685019206146
    Train Recall Score : 0.9568
    Train F1 Score : 0.8506401137980085
                  precision    recall  f1-score   support
    
             0.0       0.92      0.62      0.74       482
             1.0       0.77      0.96      0.85       625
    
        accuracy                           0.81      1107
       macro avg       0.84      0.79      0.80      1107
    weighted avg       0.83      0.81      0.80      1107
    
    

The result above shows the metrics for train dataset. All metrics is fairly good and f1 score shows a good percent. 

But let’s compare it with the test metrics:


```python
#print confusion matrix
y_pred = SVM_QQQ.predict(test_X_scaled)
confusion_matrix_QQQ = confusion_matrix(y_test_QQQ,y_pred)
print("Confusion Matrix - SVM Model")
print(confusion_matrix_QQQ)
print(classification_report(y_test_QQQ,y_pred))
```

    Confusion Matrix - SVM Model
    [[ 60  96]
     [ 39 175]]
                  precision    recall  f1-score   support
    
             0.0       0.61      0.38      0.47       156
             1.0       0.65      0.82      0.72       214
    
        accuracy                           0.64       370
       macro avg       0.63      0.60      0.60       370
    weighted avg       0.63      0.64      0.62       370
    
    


```python
print('Accuracy Score : ' + str(accuracy_score(y_test_QQQ,y_pred)))
print('Precision Score : ' + str(precision_score(y_test_QQQ,y_pred)))
print('Recall Score : ' + str(recall_score(y_test_QQQ,y_pred)))
print('F1 Score : ' + str(f1_score(y_test_QQQ,y_pred)))
```

    Accuracy Score : 0.6351351351351351
    Precision Score : 0.6457564575645757
    Recall Score : 0.8177570093457944
    F1 Score : 0.7216494845360825
    

The result above shows, there is a huge underfitted status as all metrics is in the test metrics are less than train metrics. As the model is uderfitted, we have to try more complex model and on the other hand increase complexity.  There are many reasons caused underfitting and one of them is we are using a simple model. 
So we will try to find the best parameters for the model by ‘GridSearchCV’. This is called tuning he hyper parameters and it struggles to find the best parameters. Then we will recheck the result again.


**Tuning the Hyper parameters**

Now we will introduce the different values for tuning the hyper parameters. As there are three major hyper parameters in SVM, such ‘C’, ‘gamma’ and ‘kernel’ we will assign the numbers for these hyper parameters:


```python
my_param_grid = {'C':[10,100,1000],'gamma':[1,0.1,0.01],'kernel':['rbf','linear','sigmoid']}
```

I have assigned ‘C’ as range of values 10, 100, 1000 from the soft margin to hard margin. Based on a default values ‘gamma’ has a default ‘scale’. So we have changed from 1 to 0.01. In the kernel changes, I will not go for ‘poly’ option as for this project that would not a good option, but we have added sigmoid and having the chance to see how model can deal with it. 

So I have put together into a function called ‘grid’ and set for 5 cross validation.


```python
grid = GridSearchCV(estimator = SVC(probability=True),param_grid = my_param_grid, refit=True,verbose=2, cv=5)
```


```python
grid.fit(train_X_scaled,y_train_QQQ)
```

    Fitting 5 folds for each of 27 candidates, totalling 135 fits
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.3s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.3s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.3s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.3s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.4s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.3s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.3s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.3s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.3s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.3s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.3s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.4s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.7s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.7s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   1.0s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.5s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.5s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.2s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.1s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.5s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   1.0s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   1.0s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.5s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.5s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.1s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.1s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.1s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.7s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.7s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.8s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.5s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.1s
    




    GridSearchCV(cv=5, estimator=SVC(probability=True),
                 param_grid={'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01],
                             'kernel': ['rbf', 'linear', 'sigmoid']},
                 verbose=2)



The best parameters run by the GridSearchCV are as follows:


```python
grid.best_params_
```




    {'C': 1000, 'gamma': 1, 'kernel': 'linear'}




```python
SVM_QQQ =SVC(C=1000,gamma=1,kernel="linear",probability=True)
SVM_QQQ.fit(train_X_scaled,y_train_QQQ)
y_pred_train = SVM_QQQ.predict(train_X_scaled)
```


```python
y_pred_train = SVM_QQQ.predict(train_X_scaled)
print('Train Accuracy Score : ' + str(accuracy_score(y_train_QQQ,y_pred_train)))
print('Train Precision Score : ' + str(precision_score(y_train_QQQ,y_pred_train)))
print('Train Recall Score : ' + str(recall_score(y_train_QQQ,y_pred_train)))
print('Train F1 Score : ' + str(f1_score(y_train_QQQ,y_pred_train)))
print(classification_report(y_train_QQQ,y_pred_train))
```

    Train Accuracy Score : 0.982836495031617
    Train Precision Score : 0.9779179810725552
    Train Recall Score : 0.992
    Train F1 Score : 0.9849086576648134
                  precision    recall  f1-score   support
    
             0.0       0.99      0.97      0.98       482
             1.0       0.98      0.99      0.98       625
    
        accuracy                           0.98      1107
       macro avg       0.98      0.98      0.98      1107
    weighted avg       0.98      0.98      0.98      1107
    
    

As we can see all metrics have been improved and they are completely different in comparison with the simple model. 

Let’s check the test dataset and compare the metrics


```python
y_pred = SVM_QQQ.predict(test_X_scaled)
print('Accuracy Score : ' + str(accuracy_score(y_test_QQQ,y_pred)))
print('Precision Score : ' + str(precision_score(y_test_QQQ,y_pred)))
print('Recall Score : ' + str(recall_score(y_test_QQQ,y_pred)))
print('F1 Score : ' + str(f1_score(y_test_QQQ,y_pred)))
print(classification_report(y_test_QQQ,y_pred))
```

    Accuracy Score : 0.9540540540540541
    Precision Score : 0.9712918660287081
    Recall Score : 0.9485981308411215
    F1 Score : 0.9598108747044917
                  precision    recall  f1-score   support
    
             0.0       0.93      0.96      0.95       156
             1.0       0.97      0.95      0.96       214
    
        accuracy                           0.95       370
       macro avg       0.95      0.96      0.95       370
    weighted avg       0.95      0.95      0.95       370
    
    

Now, we can see the test metrics have been improved. Although it is still a bit underfitted but the numbers are approximately close together.


```python
ns_probs = [0 for _ in range(len(y_test_QQQ))]
# predict probabilities
lr_probs = SVM_QQQ.predict(test_X_scaled)
# keep probabilities for the positive outcome only
lr_probs = lr_probs
# calculate scores
ns_auc = roc_auc_score(y_test_QQQ, ns_probs)
lr_auc = roc_auc_score(y_test_QQQ, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test_QQQ, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test_QQQ, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Support Vector Machine, C = 1000')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```

    No Skill: ROC AUC=0.500
    Logistic: ROC AUC=0.955
    


<img src="/img/SVM_Prediction_99_1.png" alt="" />               

    


Now for computing metrics for within the class we have to define two data frame based on ‘Sign’ variable.
(For localized time, my system popped up an error that it is already converted then I converted to comment. If you need it please make it uncomment and run the code) 



```python
df1 = QQQ_features[-len(y_test_QQQ):]
df1['Signal'] = y_pred 
df1['Strategy'] = df1['ret_0'] * df1['Signal'].fillna(0)

#df1.index = df1.index.tz_localize('utc')
df1[['Adj Close', 'ret_0', 'Signal', 'Strategy']].tail(5)
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
      <th>Adj Close</th>
      <th>ret_0</th>
      <th>Signal</th>
      <th>Strategy</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-21 00:00:00-05:00</th>
      <td>226.743179</td>
      <td>-0.019396</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2020-02-24 00:00:00-05:00</th>
      <td>217.999130</td>
      <td>-0.039327</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2020-02-25 00:00:00-05:00</th>
      <td>212.071365</td>
      <td>-0.027568</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2020-02-26 00:00:00-05:00</th>
      <td>213.164337</td>
      <td>0.005141</td>
      <td>1.0</td>
      <td>0.005141</td>
    </tr>
    <tr>
      <th>2020-02-27 00:00:00-05:00</th>
      <td>202.490387</td>
      <td>-0.051371</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sign_1=df1[df1['Sign']==1]
df_sign_0=df1[df1['Sign']==0]
```


```python
print(confusion_matrix(df_sign_1['Sign'],df_sign_1['Signal']))
print(classification_report(df_sign_1['Sign'],df_sign_1['Signal']))
```

    [[  0   0]
     [ 11 203]]
                  precision    recall  f1-score   support
    
             0.0       0.00      0.00      0.00         0
             1.0       1.00      0.95      0.97       214
    
        accuracy                           0.95       214
       macro avg       0.50      0.47      0.49       214
    weighted avg       1.00      0.95      0.97       214
    
    

We can see the confusion matrix and classification report for ‘Sign’ equals to one and ‘Signal’ is the prediction. Since the 1 means the trend is going up, there is an accuracy equals 95% which is very high, but this not an issue because the downward trend is around this percentage. The report for downtrend is:


```python
print(confusion_matrix(df_sign_0['Sign'],df_sign_0['Signal']))
print(classification_report(df_sign_0['Sign'],df_sign_0['Signal']))
```

    [[150   6]
     [  0   0]]
                  precision    recall  f1-score   support
    
             0.0       1.00      0.96      0.98       156
             1.0       0.00      0.00      0.00         0
    
        accuracy                           0.96       156
       macro avg       0.50      0.48      0.49       156
    weighted avg       1.00      0.96      0.98       156
    
    

# **6 & 7 - Model Evaluation for selected features**

Next, we will apply the exact computation above for the features selected by univariate process.


```python
cols_selected=['MOM5', 'MOM7','MOM13', 'MOM21','EWMA5'] 
```


```python
X_train_QQQ,X_test_QQQ,y_train_QQQ,y_test_QQQ = train_test_split(QQQ_features[cols_selected], QQQ_features['Sign'],
                                                                 test_size=0.25, shuffle=False)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_X_scaled = scaler.fit_transform(X_train_QQQ)
test_X_scaled = scaler.transform(X_test_QQQ)

train_X_scaled = pd.DataFrame(train_X_scaled,columns=X_train_QQQ.columns)
test_X_scaled = pd.DataFrame(test_X_scaled,columns=X_train_QQQ.columns)

```


```python
SVM_QQQ =SVC(probability=True)
SVM_QQQ.fit(train_X_scaled,y_train_QQQ)
```




    SVC(probability=True)




```python
y_pred_train = SVM_QQQ.predict(train_X_scaled)
print('Train Accuracy Score : ' + str(accuracy_score(y_train_QQQ,y_pred_train)))
print('Train Precision Score : ' + str(precision_score(y_train_QQQ,y_pred_train)))
print('Train Recall Score : ' + str(recall_score(y_train_QQQ,y_pred_train)))
print('Train F1 Score : ' + str(f1_score(y_train_QQQ,y_pred_train)))
print(classification_report(y_train_QQQ,y_pred_train))
```

    Train Accuracy Score : 0.6657633242999097
    Train Precision Score : 0.6679841897233202
    Train Recall Score : 0.8112
    Train F1 Score : 0.7326589595375722
                  precision    recall  f1-score   support
    
             0.0       0.66      0.48      0.55       482
             1.0       0.67      0.81      0.73       625
    
        accuracy                           0.67      1107
       macro avg       0.66      0.64      0.64      1107
    weighted avg       0.66      0.67      0.65      1107
    
    

This is for training dataset. We can see accuracy is lower than the simple model of all variables. Now we check the metrics for test dataset


```python
#print confusion matrix
y_pred = SVM_QQQ.predict(test_X_scaled)
confusion_matrix_QQQ = confusion_matrix(y_test_QQQ,y_pred)
print("Confusion Matrix - SVM Model")
print(confusion_matrix_QQQ)
print(classification_report(y_test_QQQ,y_pred))
```

    Confusion Matrix - SVM Model
    [[ 88  68]
     [ 76 138]]
                  precision    recall  f1-score   support
    
             0.0       0.54      0.56      0.55       156
             1.0       0.67      0.64      0.66       214
    
        accuracy                           0.61       370
       macro avg       0.60      0.60      0.60       370
    weighted avg       0.61      0.61      0.61       370
    
    


```python
print('Accuracy Score : ' + str(accuracy_score(y_test_QQQ,y_pred)))
print('Precision Score : ' + str(precision_score(y_test_QQQ,y_pred)))
print('Recall Score : ' + str(recall_score(y_test_QQQ,y_pred)))
print('F1 Score : ' + str(f1_score(y_test_QQQ,y_pred)))
```

    Accuracy Score : 0.6108108108108108
    Precision Score : 0.6699029126213593
    Recall Score : 0.6448598130841121
    F1 Score : 0.6571428571428571
    

As we can see the model is also underfitted, but not like the first model with all variables. The metrics are close together. Now I will apply the grid search for feature selected model and then we can compare the result.

**Tuning the Hyper parameters**


```python
my_param_grid = {'C':[10,100,1000],'gamma':[1,0.1,0.01],'kernel':['rbf','linear','sigmoid']}
```


```python
grid = GridSearchCV(estimator = SVC(probability=True),param_grid = my_param_grid, refit=True,verbose=2, cv=5)
```


```python
grid.fit(train_X_scaled,y_train_QQQ)
```

    Fitting 5 folds for each of 27 candidates, totalling 135 fits
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .......................C=10, gamma=1, kernel=linear; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=10, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END .....................C=10, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ....................C=10, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ....................C=10, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=10, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.2s
    [CV] END ......................C=100, gamma=1, kernel=linear; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END .....................C=100, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.2s
    [CV] END ....................C=100, gamma=0.1, kernel=linear; total time=   0.1s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ...................C=100, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.2s
    [CV] END ...................C=100, gamma=0.01, kernel=linear; total time=   0.1s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=100, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.9s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.8s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.9s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.9s
    [CV] END ........................C=1000, gamma=1, kernel=rbf; total time=   0.7s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.6s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.6s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.6s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.6s
    [CV] END .....................C=1000, gamma=1, kernel=linear; total time=   0.4s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ....................C=1000, gamma=1, kernel=sigmoid; total time=   0.1s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END ......................C=1000, gamma=0.1, kernel=rbf; total time=   0.3s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.6s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.6s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   1.1s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.8s
    [CV] END ...................C=1000, gamma=0.1, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.3s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.3s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.2s
    [CV] END ..................C=1000, gamma=0.1, kernel=sigmoid; total time=   0.3s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.3s
    [CV] END .....................C=1000, gamma=0.01, kernel=rbf; total time=   0.2s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.6s
    [CV] END ..................C=1000, gamma=0.01, kernel=linear; total time=   0.4s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    [CV] END .................C=1000, gamma=0.01, kernel=sigmoid; total time=   0.2s
    




    GridSearchCV(cv=5, estimator=SVC(probability=True),
                 param_grid={'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01],
                             'kernel': ['rbf', 'linear', 'sigmoid']},
                 verbose=2)



The best parameters run by the GridSearchCV are as follows:


```python
grid.best_params_
```




    {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}




```python
SVM_QQQ =SVC(C=10,gamma=0.1,kernel="rbf",probability=True)
SVM_QQQ.fit(train_X_scaled,y_train_QQQ)
y_pred_train = SVM_QQQ.predict(train_X_scaled)
```


```python
y_pred_train = SVM_QQQ.predict(train_X_scaled)
print('Train Accuracy Score : ' + str(accuracy_score(y_train_QQQ,y_pred_train)))
print('Train Precision Score : ' + str(precision_score(y_train_QQQ,y_pred_train)))
print('Train Recall Score : ' + str(recall_score(y_train_QQQ,y_pred_train)))
print('Train F1 Score : ' + str(f1_score(y_train_QQQ,y_pred_train)))
print(classification_report(y_train_QQQ,y_pred_train))
```

    Train Accuracy Score : 0.6504065040650406
    Train Precision Score : 0.6461916461916462
    Train Recall Score : 0.8416
    Train F1 Score : 0.7310632383599723
                  precision    recall  f1-score   support
    
             0.0       0.66      0.40      0.50       482
             1.0       0.65      0.84      0.73       625
    
        accuracy                           0.65      1107
       macro avg       0.65      0.62      0.62      1107
    weighted avg       0.65      0.65      0.63      1107
    
    

The parameters which have been found by grid search here is less than those in the model for all variables.	
There is an improvement in the metrics by grid search approach and let’s check the result with test dataset


```python
y_pred = SVM_QQQ.predict(test_X_scaled)
print('Accuracy Score : ' + str(accuracy_score(y_test_QQQ,y_pred)))
print('Precision Score : ' + str(precision_score(y_test_QQQ,y_pred)))
print('Recall Score : ' + str(recall_score(y_test_QQQ,y_pred)))
print('F1 Score : ' + str(f1_score(y_test_QQQ,y_pred)))
print(classification_report(y_test_QQQ,y_pred))
```

    Accuracy Score : 0.6486486486486487
    Precision Score : 0.6653543307086615
    Recall Score : 0.7897196261682243
    F1 Score : 0.7222222222222223
                  precision    recall  f1-score   support
    
             0.0       0.61      0.46      0.52       156
             1.0       0.67      0.79      0.72       214
    
        accuracy                           0.65       370
       macro avg       0.64      0.62      0.62       370
    weighted avg       0.64      0.65      0.64       370
    
    


```python
ns_probs = [0 for _ in range(len(y_test_QQQ))]
# predict probabilities
lr_probs = SVM_QQQ.predict(test_X_scaled)
# keep probabilities for the positive outcome only
lr_probs = lr_probs
# calculate scores
ns_auc = roc_auc_score(y_test_QQQ, ns_probs)
lr_auc = roc_auc_score(y_test_QQQ, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test_QQQ, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test_QQQ, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Support Vector Machine, C = 1000')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
```

    No Skill: ROC AUC=0.500
    Logistic: ROC AUC=0.622
    


<img src="/img/SVM_Prediction_126_1.png" alt="" />                   

    


 So, we will generate confusion matrix and classification report for selected features within class and the output is as follows: 


```python
# Create a new dataframe to subsume outsample data
df1 = QQQ_features[-len(y_test_QQQ):]

# Predict the signal and store in predicted signal column
df1['Signal'] = y_pred
    
# Calculate the strategy returns
df1['Strategy'] = df1['ret_0'] * df1['Signal'].shift(1).fillna(0)

# Localize index for pyfolio
#df1.index = df1.index.tz_localize('utc')

# Check the output
df1[['Adj Close', 'ret_0', 'Signal', 'Strategy']].tail(10)
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
      <th>Adj Close</th>
      <th>ret_0</th>
      <th>Signal</th>
      <th>Strategy</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-13 00:00:00-05:00</th>
      <td>230.386490</td>
      <td>-0.001281</td>
      <td>1.0</td>
      <td>-0.001281</td>
    </tr>
    <tr>
      <th>2020-02-14 00:00:00-05:00</th>
      <td>231.046204</td>
      <td>0.002859</td>
      <td>1.0</td>
      <td>0.002859</td>
    </tr>
    <tr>
      <th>2020-02-18 00:00:00-05:00</th>
      <td>231.134857</td>
      <td>0.000384</td>
      <td>1.0</td>
      <td>0.000384</td>
    </tr>
    <tr>
      <th>2020-02-19 00:00:00-05:00</th>
      <td>233.350357</td>
      <td>0.009540</td>
      <td>1.0</td>
      <td>0.009540</td>
    </tr>
    <tr>
      <th>2020-02-20 00:00:00-05:00</th>
      <td>231.184082</td>
      <td>-0.009327</td>
      <td>1.0</td>
      <td>-0.009327</td>
    </tr>
    <tr>
      <th>2020-02-21 00:00:00-05:00</th>
      <td>226.743164</td>
      <td>-0.019396</td>
      <td>0.0</td>
      <td>-0.019396</td>
    </tr>
    <tr>
      <th>2020-02-24 00:00:00-05:00</th>
      <td>217.999176</td>
      <td>-0.039327</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2020-02-25 00:00:00-05:00</th>
      <td>212.071365</td>
      <td>-0.027568</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2020-02-26 00:00:00-05:00</th>
      <td>213.164352</td>
      <td>0.005141</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-02-27 00:00:00-05:00</th>
      <td>202.490387</td>
      <td>-0.051371</td>
      <td>0.0</td>
      <td>-0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = QQQ_features[-len(y_test_QQQ):]
df1['Signal'] = y_pred 
df1['Strategy'] = df1['ret_0'] * df1['Signal'].fillna(0)

#df1.index = df1.index.tz_localize('utc')
df1[['Adj Close', 'ret_0', 'Signal', 'Strategy']].tail(5)
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
      <th>Adj Close</th>
      <th>ret_0</th>
      <th>Signal</th>
      <th>Strategy</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-21 00:00:00-05:00</th>
      <td>226.743179</td>
      <td>-0.019396</td>
      <td>0.0</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>2020-02-24 00:00:00-05:00</th>
      <td>217.999130</td>
      <td>-0.039327</td>
      <td>0.0</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>2020-02-25 00:00:00-05:00</th>
      <td>212.071365</td>
      <td>-0.027568</td>
      <td>0.0</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>2020-02-26 00:00:00-05:00</th>
      <td>213.164337</td>
      <td>0.005141</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-27 00:00:00-05:00</th>
      <td>202.490387</td>
      <td>-0.051371</td>
      <td>0.0</td>
      <td>-0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sign_1=df1[df1['Sign']==1]
df_sign_0=df1[df1['Sign']==0]
```


```python
print(confusion_matrix(df_sign_1['Sign'],df_sign_1['Signal']))
print(classification_report(df_sign_1['Sign'],df_sign_1['Signal']))
```

    [[  0   0]
     [ 45 169]]
                  precision    recall  f1-score   support
    
             0.0       0.00      0.00      0.00         0
             1.0       1.00      0.79      0.88       214
    
        accuracy                           0.79       214
       macro avg       0.50      0.39      0.44       214
    weighted avg       1.00      0.79      0.88       214
    
    


```python
print(confusion_matrix(df_sign_0['Sign'],df_sign_0['Signal']))
print(classification_report(df_sign_0['Sign'],df_sign_0['Signal']))
```

    [[71 85]
     [ 0  0]]
                  precision    recall  f1-score   support
    
             0.0       1.00      0.46      0.63       156
             1.0       0.00      0.00      0.00         0
    
        accuracy                           0.46       156
       macro avg       0.50      0.23      0.31       156
    weighted avg       1.00      0.46      0.63       156
    
    

In summary, in this project we started by loading data and creating our features we need to predict the positive movement. Then we examined some primary variables with our target variable called ‘Sign’ and see how the graphs and numbers behave when we changed the hyper parameters and variables on train and validation data sets. Then we decided to create two models on with all variables and the other with selected features which we extracted from univariate feature selection. Then we ran the model with default parameters and then optimized that by grid searching and refitted again and took the metrics. We saw this path for both model and finally we get the confusion matrix and classification reports for uptrends and downtrends movements which we can see in the table below. 

**References:**

**CQF Lectures and Tutorial Module 4 and 5**

**P. Wilmott - Machine Learning: An Applied Mathematics Introduction**

**Python Data Science Handbook by Jake VanderPlas** 



```python

```
