---
title: Market Risk - Part I :VaR
author: ''
date: '2023-01-02'
slug: value-at-risk
categories: []
tags: [Finance]
subtitle: ''
summary: 'VaR and CVaR calculation and solved an EWMA case '
authors: []
lastmod: '2023-01-02T12:46:51+01:00'
featured: no
image:
  placement: 6 
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

# Market Risk - Part I : Value at Risk 

In this post I will talk about one the important and common tools of measuring financial risk. Value at Risk or briefly VaR is the most important metrics that is used to measure the risk associated with the financial positions or a portfolio of financial instruments. This concept for the first time was purposed by J.P Morgan executives and they were following a common and vital question, ***what is the maximum loss or more truley the expected maximum loss in an investment***. Then this method has became well-know Basel framework recommend financial institution to apply this approach for their risk measures.

Value at Risk gives an estimated loss amount over a holiding period through a confidence interval.
Generally it has three main components:
  - A standard deviation as a proxy of downside risk or loss level 
  - A time horizon that it is fixed and risk is assessed over that
  - A confidence interval 

A common phrase for expressing VaR is ***I am 95% confident that for 1 week ( or daily and etc.), the maximum loss will not exceed € A*** or alternatively one can express ***There is a 5% chance that the loss will exceed € A or more*** in terms of confidence and maximum loss levels respectively.

Value at Risk can be applied by three approaches:

  - Variance-Covariance or Parametric method
  - Historical approach
  - Monte Carlo approach




```python
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import t
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
# Plot settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [24.0, 8.0]
matplotlib.rcParams['font.size'] = 5
#matplotlib.rcParams['lines.linewidth'] = 2.0
#matplotlib.rcParams['grid.color'] = 'black'

import yfinance as yf

# Data manipulation
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot
import random
from tabulate import tabulate
```


```python
initial = '2020-07-14'
final = '2022-07-14'
df = yf.download('AMZN,IBM,AAPL,CSCO' , start = initial, end = final, progress=False, as_json=False, order='ascending', interval='1d')
```


```python
df = df['Adj Close']
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>CSCO</th>
      <th>IBM</th>
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
      <th>2020-07-14</th>
      <td>95.568726</td>
      <td>154.199997</td>
      <td>43.176174</td>
      <td>101.759117</td>
    </tr>
    <tr>
      <th>2020-07-15</th>
      <td>96.225983</td>
      <td>150.443497</td>
      <td>43.306839</td>
      <td>103.784187</td>
    </tr>
    <tr>
      <th>2020-07-16</th>
      <td>95.041908</td>
      <td>149.994995</td>
      <td>42.728172</td>
      <td>104.636398</td>
    </tr>
    <tr>
      <th>2020-07-17</th>
      <td>94.849915</td>
      <td>148.098495</td>
      <td>43.633511</td>
      <td>105.564545</td>
    </tr>
    <tr>
      <th>2020-07-20</th>
      <td>96.848770</td>
      <td>159.841995</td>
      <td>43.838852</td>
      <td>106.627708</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate daily returns
returns = df.pct_change().dropna()
returns.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>CSCO</th>
      <th>IBM</th>
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
      <th>2020-07-15</th>
      <td>0.006877</td>
      <td>-0.024361</td>
      <td>0.003026</td>
      <td>0.019901</td>
    </tr>
    <tr>
      <th>2020-07-16</th>
      <td>-0.012305</td>
      <td>-0.002981</td>
      <td>-0.013362</td>
      <td>0.008211</td>
    </tr>
    <tr>
      <th>2020-07-17</th>
      <td>-0.002020</td>
      <td>-0.012644</td>
      <td>0.021188</td>
      <td>0.008870</td>
    </tr>
    <tr>
      <th>2020-07-20</th>
      <td>0.021074</td>
      <td>0.079295</td>
      <td>0.004706</td>
      <td>0.010071</td>
    </tr>
    <tr>
      <th>2020-07-21</th>
      <td>-0.013802</td>
      <td>-0.018315</td>
      <td>0.001064</td>
      <td>-0.002453</td>
    </tr>
  </tbody>
</table>
</div>



So, before going to calculate VaR in these three methods, consider the return for each stock in a figure


```python
fig, ax = plt.subplots(1,len(returns.columns),sharey=True)
label , color = df.columns, ['green', 'red', 'cornflowerblue', 'orange', 'white']

for i in range(len(returns.columns)):
    ax[i].plot(returns.iloc[:,i], label=label[i], color= color[i])
    ax[i].axhline(y=0, color='k',linestyle='--')
    ax[i].legend(loc=4)
fig.suptitle('Daily Returns')
plt.show()
```


 <img src="/img/Var_Problem_8_0.png" alt="" />    

    


<font size="4">__Variance - Covariance or Parametric Approach__</font>

The Variance - Covariance approach mostly assumes that the returns are normally distributed.ALthough there is alternative distribution such as t-distribution, I will talk about in Monte Carlo section.
In this method we first calculate the mean and standard deviation of the returns to derive the risk metric. 
Based on the assumption of normality, we can generalise: 
 - **confidence interval 90%**  $\rightarrow$  ***VaR*** = $\mu - 1.29 * \sigma $
 - **confidence interval 95%**  $\rightarrow$  ***VaR*** = $\mu - 1.64 * \sigma $
 - **confidence interval 95%**  $\rightarrow$  ***VaR*** = $\mu - 2.33 * \sigma $



```python
def VaR(dataframe):
    var = pd.DataFrame()
    
    for i in [90,95,99]:
        for j in range(len(dataframe.columns)):
            var.loc[i,j] = 100 * norm.ppf(1-i/100,dataframe.iloc[:,j].mean(),dataframe.iloc[:,j].std())
            
    var.columns = dataframe.columns
    return var
        
```


```python
VAR = VaR(returns)
VAR
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>CSCO</th>
      <th>IBM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>-2.495749</td>
      <td>-2.974684</td>
      <td>-2.015413</td>
      <td>-1.863421</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-3.232597</td>
      <td>-3.806596</td>
      <td>-2.589214</td>
      <td>-2.410374</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-4.614801</td>
      <td>-5.367125</td>
      <td>-3.665569</td>
      <td>-3.436367</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's assume that we have 1,000 shares of CSCO's stock on 2022-07-14. What is the maximum loss next day with a confidence level of 99%?


```python
shares = 1000
price = df['CSCO'].iloc[-1]
position = shares * price
VAR_99 = VAR.loc[99,'CSCO']/100

CSCO_VaR= position * VAR_99
CSCO_VaR_5days = position * VAR_99* np.sqrt(5)
print(f'CSCO Holding Value :{position}')
print(f'CSCO VaR at 99% confidence level is: {CSCO_VaR}')
print(f'CSCO VaR at 99% confidence level is for 5 days: {CSCO_VaR_5days}')

```

    CSCO Holding Value :42307.02590942383
    CSCO VaR at 99% confidence level is: -1550.7933638301922
    CSCO VaR at 99% confidence level is for 5 days: -3467.6793805798734
    


```python
# Scaled VaR over different time horizon
# plt.figure()
plt.plot(range(100),[-100*VAR_99*np.sqrt(x) for x in range(100)], color='red')
plt.xlabel('Horizon')
plt.ylabel('Var 99 (%)')
plt.title('VaR_99 Scaled by Time');
```


<img src="/img/Var_Problem_15_0.png" alt="" />  

    



```python
ret_mean = returns.mean()
weights = np.random.random(len(returns.columns))
weights /=np.sum(weights)
weights = pd.DataFrame(weights).T
weights.columns = returns.columns
weights
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>CSCO</th>
      <th>IBM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.478569</td>
      <td>0.178027</td>
      <td>0.160937</td>
      <td>0.182466</td>
    </tr>
  </tbody>
</table>
</div>



**Portfolio Return**


```python
port_ret = np.dot(ret_mean,weights.T).flatten()[0]
port_ret
```




    0.0005581899979238245



**Portfolio Volatility**


```python
port_stdev = np.sqrt(multi_dot([weights,returns.cov(),weights.T])).flatten()[0]
port_stdev
```




    0.014874279669923446



**Portfolio VaR in 95%**

Let's now compare the portfolio VaR numbers with that of the individual stocks


```python
random.seed(42)
portpos = 0
for stock in returns.columns:
    pos = df[stock].iloc[-1] * 1000 * weights[stock][0]
    p_var = VaR(returns)[stock].iloc[1]
    
    print(f'{stock} Holding Value: {pos:0.8}')
    print(f'{stock} VaR at 95% confidence level: {p_var:0.8}%')
    print()
    
    portpos+= pos
    
print(f'Portfolio Holding Value: {portpos:0.8}')
print(f'Portfolio VaR at 95% confidence level {norm.ppf(1-0.95,port_ret,port_stdev)*100:0.3}%')
    
```

    AAPL Holding Value: 69415.338
    AAPL VaR at 95% confidence level: -3.2325966%
    
    AMZN Holding Value: 19654.199
    AMZN VaR at 95% confidence level: -3.8065963%
    
    CSCO Holding Value: 6808.7815
    CSCO VaR at 95% confidence level: -2.5892141%
    
    IBM Holding Value: 24427.993
    IBM VaR at 95% confidence level: -2.4103741%
    
    Portfolio Holding Value: 120306.31
    Portfolio VaR at 95% confidence level -2.39%
    

As we can see the VaR for the current portfolio is **2.39%** which is much less than the individual VaRs. This indicates the effect of diversification by having different stocks in the basket

As we have seen the calculation of Variance - Covariance approach, there are some advantages and disadvantages of applying this model. Although the advantages are such that the calculation of model is easy and there is no need to a large numbers of samples, however the disadvantages such as normall distribution assumption, nonlinear patterns and covariance matrix, make enough sense for analysts to skip applying that.
There is also a naive way to eliminate the **'normal distribution'** assumption called ***Historical VaR .***

<font size="4">__Historical VaR Approach__</font>

Historical VaR is an alternative way when there is no assumption about distribution. Asset returns do not necessarily follow a normal distribution. This method uses historical data where returns are sorted in ascending order to calculate maximum possible loss for a given confidence level. 


```python
Hist_VaR_90 = returns['IBM'].quantile(0.10)
Hist_VaR_95 = returns['IBM'].quantile(0.05)
Hist_Var_99 = returns['IBM'].quantile(0.01)
```


```python
htable =[['90%',Hist_VaR_90],['95%',Hist_VaR_95],['99%',Hist_Var_99]]
header = ['Confidence Level', 'Value At Risk']
print(tabulate(htable,headers=header))
```

    Confidence Level      Value At Risk
    ------------------  ---------------
    90%                      -0.0144039
    95%                      -0.0196433
    99%                      -0.0391651
    

<p style='text-align: justify;'>  As it has been shown, the calculation of historical VaR is fairly easy and it does not need to follow the normal distribution and it is suitable for non-normal assumption. However this method requires a large sample of data and it laso needs a powerful computating power as we need to store and work with all tickres data. One important deficit that both methods, parametric and historical VaR do not provide for us is, there is no different scenarios to and their tendencies. This problem can be solved by a powerfull method in finance called "Monte Carlo simulation" </p>

<font size="4">__Monte Carlo simulation Approach__</font>

***Monte Carlo simulation*** is a popular numerical technique in finance and quatitative analysts are applying it when there is no colsed-form solution.

The Monte Carlo simulation approach has a number of similarities to historical simulation. It allows us to use actual historical distributions rather than having to assume normal returns. As returns are assumed to follow a normal distribution, we could generate n simulated returns with the same mean and standard deviation (derived from the daily returns) and then sorted in ascending order to calculate maximum possible loss for a given confidence level.

In most cases, Monte Carlo is simulated by normal distribution, as we know this subject as a negative point and I indicated in the first method. In finance an alternative distribution for normal is ***t - distribution***  ***(or t - Student distribution )*** as it has fatter tails than normal distribution and more values are located in the tails of t-distribution. Sometimes quantitative analysts use ***uniform distribution*** and then generating the paths and Monte Carlo simulation.


Now, lets consider ***Apple and IBM*** prices and see what scenarios we face with Monte Carlo simulation based on normal distributions.


```python
df_2 = pd.melt(df,value_vars=['AAPL','AMZN','IBM','CSCO'])
df_2.rename(columns={'variable':'symbol','value':'close'},inplace=True)
```


```python
def MC_Algo(dataframe,ticker,t_intervals,iterations):
    from scipy.stats import t
    import scipy.stats
    dataframe = df_2[df_2['symbol']==ticker]
    log_returns = np.log(1 + dataframe.close.pct_change())
    u = log_returns.mean()
    var = log_returns.var() 
    drift = u - (0.5 * var) 
    stdev = log_returns.std() 

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    t_intervals = t_intervals 
    iterations = iterations
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

    S0 = dataframe.close.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0


    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    
    price_list = pd.DataFrame(price_list)
    price_list['close'] = price_list[0]
    price_list.head()
    
    close = dataframe.close
    close = pd.DataFrame(close)
    frames = [close, price_list]
    monte_carlo_forecast = pd.concat(frames)
    
    monte_carlo = monte_carlo_forecast.iloc[:,:].values
    plt.figure(figsize=(17,8))
    plt.plot(monte_carlo)
    plt.title( 'Monte Carlo Simulation with {} interval and {} iteration for stock {}'.format(t_intervals,iterations,ticker))
    plt.show()



```


```python
MC_Algo(df_2,'AAPL',250,10)
```


<img src="/img/Var_Problem_36_0.png" alt="" />      

    



```python
MC_Algo(df_2,'IBM',500,100)
```


 <img src="/img/Var_Problem_37_0.png" alt="" />    

    


As I mentioned befor, an alternative and more percise way is ***t - distribution***. For the first step let's compare which distribution is better for 'IBM' stock. 
Note that, for generating random numbers based on t distribution, we need to have a component called ***Degree of freedom***. We can extract this item by fitting t - distribution on retruns and apply qq plot to see the differences on that.


```python
scipy.stats.probplot(returns['IBM'], dist=scipy.stats.norm, plot=plt.figure().add_subplot(111))
plt.title("Normal probability plot of IBM daily ", weight="bold");
```


 <img src="/img/Var_Problem_39_0.png" alt="" />        

    



```python
tdf, tmean, tsigma = scipy.stats.t.fit(returns['IBM'])
scipy.stats.probplot(returns['AAPL'], dist=scipy.stats.t, sparams=(tdf, tmean, tsigma), plot=plt.figure().add_subplot(111))
plt.title("Student probability plot of IBM daily", weight="bold");
```


 <img src="/img/Var_Problem_40_0.png" alt="" />           

    


As we can see the ***t - distribution*** fits better than normal distribution and this is what I have mentioned about the fatter tails of ***t distribution*** in comparison with ***normal distribution***

***Value at Risk with Monte Carlo simulation on Portfolio*** 


```python
np.random.seed(42)

# Number of simulations
n_sims = 5000

# Simulate returns and sort
sim_returns = np.random.normal(port_ret, port_stdev, n_sims)

# Use percentile function for MCVaR
MCVaR_90 = np.percentile(sim_returns,10)
MCVaR_95 = np.percentile(sim_returns, 5)
MCVaR_99 = np.percentile(sim_returns,1)
```


```python
mctable = [['90%', MCVaR_90],['95%', MCVaR_95],['99%', MCVaR_99]]
print(tabulate(mctable,headers=header))
```

    Confidence Level      Value At Risk
    ------------------  ---------------
    90%                      -0.0183232
    95%                      -0.0234435
    99%                      -0.03459
    


```python
weighted_returns = weights.values * returns
```


```python
port_rets = weighted_returns.sum(axis=1)
port_rets.hist(bins=40,density=True,alpha=0.5,color='blue')
plt.axvline(x= -0.0195903,color='yellow')
plt.axvline(x= -0.0249358,color='orange')
plt.axvline(x= -0.0365724,color='red')
plt.figtext(0.30, 0.8, "q(0.99): {:.3}".format(-0.0365724),weight="bold",fontsize='x-large')
plt.title("Fitting Normal - distribution for portfolio retrun", weight="bold")
```




    Text(0.5, 1.0, 'Fitting Normal - distribution for portfolio retrun')




<img src="/img/Var_Problem_46_1.png" alt="" />               

    



```python
np.random.seed(42)

# Number of simulations
n_sims = 5000

# Simulate returns and sort
tdf, tmean, tsigma = scipy.stats.t.fit(port_rets)
sim_returns = np.random.standard_t(tdf, n_sims)

# Use percentile function for MCVaR
MCVaR_90 = np.percentile(sim_returns,10)/100
MCVaR_95 = np.percentile(sim_returns, 5)/100
MCVaR_99 = np.percentile(sim_returns,1)/100
```


```python
mctable = [['90%', MCVaR_90],['95%', MCVaR_95],['99%', MCVaR_99]]
print(tabulate(mctable,headers=header))
```

    Confidence Level      Value At Risk
    ------------------  ---------------
    90%                      -0.0143352
    95%                      -0.0195043
    99%                      -0.0318826
    


```python
support = np.linspace(port_rets.min(), port_rets.max(), 100)
tdf, tmean, tsigma = scipy.stats.t.fit(port_rets)
port_rets.hist(bins=40, density=True, histtype="stepfilled", alpha=0.5,color='blue');
plt.plot(support, scipy.stats.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-")
plt.axvline(x= -0.0143352,color='yellow')
plt.axvline(x= -0.0195043,color='orange')
plt.axvline(x= -0.0318826,color='red')
plt.title("Fitting t - distribution for portfolio retrun", weight="bold");
```


 <img src="/img/Var_Problem_49_0.png" alt="" />    

    


<font size="4">__Expected Short Fall or Conditional Value at Risk__</font>

VaR is a reasonable measure of risk if assumption of normality holds. Else, we might underestimate the risk if we observe a fat tail or overestimate the risk if tail is thinner. Expected shortfall or Conditional Value at Risk - ***CVaR*** - is an estimate of expected shortfall sustained in the worst 1 - x% of scenarios. It is defined as the average loss based on the returns that are lower than the VaR threshold. Assume that we have n return observations, then the expected shortfall is

\begin{equation*}
CVaR = \frac{1}{n} * \sum_{i=1}^n R_i [R\leq hVaR_c] 
\end{equation*}

where,  ***R***  is returns,  ***hVaR***  is historical ***VaR*** and  ***c***  is the confidence level.

Let's try to see the expected short fall measure on IBM stock


```python
# Calculate CVar
CVaR_90 = returns['IBM'][returns['IBM']<=Hist_VaR_90].mean()
CVaR_95 = returns['IBM'][returns['IBM']<=Hist_VaR_95].mean()
CVaR_99 = returns['IBM'][returns['IBM']<=Hist_Var_99].mean()
```


```python
ctable = [['90%', CVaR_90],['95%', CVaR_95],['99%', CVaR_99] ]
cheader = ['Confidence Level', 'Conditional Value At Risk']
print(tabulate(ctable,headers=cheader))
```

    Confidence Level      Conditional Value At Risk
    ------------------  ---------------------------
    90%                                  -0.025394
    95%                                  -0.0337696
    99%                                  -0.0641326
    

<font size="4">__Rmetrics or EWMA model__</font>

As a market risk analyst, each day we calculate ***VaR*** from the available priordata. Then, we wait ten days to compare our prediction value ***VaRt−10*** to the realised return and check
if the prediction about the worst loss was breached. You are given a dataset with Closing Prices.
We implement VaR backtesting by computing ***99%/10day Value at Risk*** using the rolling window of 21
returns to compute σ. The report the percentage of ***VaR*** breaches and number of consecutive
breaches are needed here.We also Provide a plot which clearly identifies breaches.

• For comparison, implement backtesting using variance forecast equation below (recompute on each
day). Rolling window of 21 remains for $\sigma^2$ (past variance) computation. The equation is known as
EWMA model, and you can check how variance forecast is done in the relevant lecture.

$\sigma^2_{t+1|t} = \lambda \sigma^2_{t|t-1} + (1 - \lambda) r^2_t $

with $\lambda$ = 0.72 value set to minimise out of sample forecasting error, and $r_t$ refers to a return.



```python
def Value_at_Risk():
    df = pd.read_csv(r"C:\Users\X550LD\Desktop\Desktop2\CQF\Data_SP500.csv", index_col=0).rename(columns={'SP500': 'close'})
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
    df['returns'] = np.log(df.close) - np.log(df.close.shift(1))
    df['mean'] = df['returns'].rolling(21).mean()
    df['stdev'] = df['returns'].rolling(21).std()
    df['VaR_99_10D'] = norm.ppf(1-0.99) * df['stdev'] * np.sqrt(10) * 100
    df['Forward_Return'] = (np.log(df.close.shift(-11)) - np.log(df.close.shift(-1))) * 100
    df = df.dropna()
    df['Breaches'] = np.where(df['Forward_Return'] < df['VaR_99_10D'], 1, 0)
    print('Percentage of Breaches: ', df['Breaches'].mean()*100)
    count = df['Breaches'].groupby((df['Breaches'] != df['Breaches'].shift()).cumsum()).cumcount()
    df['Breachers_Conc'] = df['Breaches'] * count
    return df
```


```python
def BackTest(df, name, y_1, y_2, y_3, y_4):
    fig = plt.figure(figsize = (10,5))
    gs = fig.add_gridspec(nrows=3, hspace=0.1)
    ax = fig.add_subplot(gs[:-1])
    ax.get_xaxis().set_visible(False)
    fig.suptitle(f'{name}')
    sns.lineplot(data=df.reset_index(), x="Date", y=y_1, ax=ax, color='blue')
    sns.lineplot(data=df.reset_index(), x="Date", y=y_2, color='orange', ax=ax, linewidth = 1)
    sns.scatterplot(x="Date", y=y_2, data=df[df[y_3] == 1].reset_index(), ax=ax, color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Return %")
    ax.legend(labels=['Return', 'Value-at-Risk', 'Breach'])
    plt.savefig(f"Question_3_{name}.png".replace(" ", "_"), format='png', transparent=False)

```


```python
Value_at_Risk()
```

    Percentage of Breaches:  2.052545155993432
    




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
      <th>close</th>
      <th>returns</th>
      <th>mean</th>
      <th>stdev</th>
      <th>VaR_99_10D</th>
      <th>Forward_Return</th>
      <th>Breaches</th>
      <th>Breachers_Conc</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-02-21</th>
      <td>1502.420044</td>
      <td>-0.006323</td>
      <td>0.000314</td>
      <td>0.005969</td>
      <td>-4.390824</td>
      <td>2.320458</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-22</th>
      <td>1515.599976</td>
      <td>0.008734</td>
      <td>0.000658</td>
      <td>0.006243</td>
      <td>-4.592630</td>
      <td>4.492768</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-25</th>
      <td>1487.849976</td>
      <td>-0.018479</td>
      <td>-0.000223</td>
      <td>0.007513</td>
      <td>-5.527232</td>
      <td>3.643065</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-26</th>
      <td>1496.939941</td>
      <td>0.006091</td>
      <td>-0.000191</td>
      <td>0.007540</td>
      <td>-5.546486</td>
      <td>2.509814</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-27</th>
      <td>1515.989990</td>
      <td>0.012646</td>
      <td>0.000499</td>
      <td>0.008028</td>
      <td>-5.905682</td>
      <td>3.154995</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-12-13</th>
      <td>2662.850098</td>
      <td>-0.000473</td>
      <td>0.001416</td>
      <td>0.004184</td>
      <td>-3.077977</td>
      <td>0.811181</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-14</th>
      <td>2652.010010</td>
      <td>-0.004079</td>
      <td>0.001332</td>
      <td>0.004279</td>
      <td>-3.148156</td>
      <td>0.744658</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-15</th>
      <td>2675.810059</td>
      <td>0.008934</td>
      <td>0.002021</td>
      <td>0.004283</td>
      <td>-3.150677</td>
      <td>0.847653</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-18</th>
      <td>2690.159912</td>
      <td>0.005348</td>
      <td>0.001887</td>
      <td>0.004122</td>
      <td>-3.032407</td>
      <td>1.573257</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-19</th>
      <td>2681.469971</td>
      <td>-0.003235</td>
      <td>0.001858</td>
      <td>0.004157</td>
      <td>-3.058284</td>
      <td>2.356995</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1218 rows × 8 columns</p>
</div>




```python
BackTest(Value_at_Risk(), "Backtesting Value at Risk", "Forward_Return", "VaR_99_10D", "Breaches", "Breachers_Conc")
```

    Percentage of Breaches:  2.052545155993432
    


<img src="/img/Var_Problem_62_1.png" alt="" />   

    


***RiskMetrics or EWMA***


```python
def EWMA(start_var: float, sqrd_rets: list, lam: float):
    
    var = [start_var]
    for i in range(1,len(sqrd_rets)):
        var.append(lam * var[i-1] + (1-lam) * sqrd_rets[i-1])
            
    return np.array(var)

def VaR_EWMA():
    df = pd.read_csv(r"C:\Users\X550LD\Desktop\Desktop2\CQF\Data_SP500.csv", index_col=0).rename(columns={'SP500': 'close'})
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
    df['returns'] = np.log(df.close) - np.log(df.close.shift(1))
    df['mean'] = df['returns'].rolling(21).mean()
    df['stdev'] = df['returns'].rolling(21).std()
    df = df.dropna()
    df['Squared_returns'] = df['returns']**2
    df['Var_Estimate'] = ewma(df.iloc[0]['stdev']**2, df['Squared_returns'].values, 0.72)
    df['Std_Estimate'] = np.sqrt(df['Var_Estimate'])
    df['VaR_99_10D_EWMA'] = norm.ppf(1-0.99) * df['Std_Estimate'] * np.sqrt(10) * 100
    df['Forward_Return'] = (np.log(df.close.shift(-11)) - np.log(df.close.shift(-1))) * 100
    df['Breaches_EWMA'] = np.where(df['Forward_Return'] < df['VaR_99_10D_EWMA'], 1, 0)
    df = df.dropna()
    print('Percentage of Breaches: ', df['Breaches_EWMA'].mean()*100)
    groupby_func = df['Breaches_EWMA'] != df['Breaches_EWMA'].shift()
    count = (df['Breaches_EWMA'].groupby(groupby_func.cumsum()).cumcount())
    df['Breaches_EWMA_Conc'] = df['Breaches_EWMA'] * count
    return df
```


```python
VaR_EWMA()
```

    Percentage of Breaches:  2.955665024630542
    




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
      <th>close</th>
      <th>returns</th>
      <th>mean</th>
      <th>stdev</th>
      <th>Squared_returns</th>
      <th>Var_Estimate</th>
      <th>Std_Estimate</th>
      <th>VaR_99_10D_EWMA</th>
      <th>Forward_Return</th>
      <th>Breaches_EWMA</th>
      <th>Breaches_EWMA_Conc</th>
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
      <th>2013-02-21</th>
      <td>1502.420044</td>
      <td>-0.006323</td>
      <td>0.000314</td>
      <td>0.005969</td>
      <td>3.998040e-05</td>
      <td>0.000036</td>
      <td>0.005969</td>
      <td>-4.390824</td>
      <td>2.320458</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-22</th>
      <td>1515.599976</td>
      <td>0.008734</td>
      <td>0.000658</td>
      <td>0.006243</td>
      <td>7.628649e-05</td>
      <td>0.000037</td>
      <td>0.006070</td>
      <td>-4.465364</td>
      <td>4.492768</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-25</th>
      <td>1487.849976</td>
      <td>-0.018479</td>
      <td>-0.000223</td>
      <td>0.007513</td>
      <td>3.414836e-04</td>
      <td>0.000048</td>
      <td>0.006920</td>
      <td>-5.090810</td>
      <td>3.643065</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-26</th>
      <td>1496.939941</td>
      <td>0.006091</td>
      <td>-0.000191</td>
      <td>0.007540</td>
      <td>3.709877e-05</td>
      <td>0.000130</td>
      <td>0.011406</td>
      <td>-8.390818</td>
      <td>2.509814</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-02-27</th>
      <td>1515.989990</td>
      <td>0.012646</td>
      <td>0.000499</td>
      <td>0.008028</td>
      <td>1.599137e-04</td>
      <td>0.000104</td>
      <td>0.010201</td>
      <td>-7.504258</td>
      <td>3.154995</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-12-13</th>
      <td>2662.850098</td>
      <td>-0.000473</td>
      <td>0.001416</td>
      <td>0.004184</td>
      <td>2.237940e-07</td>
      <td>0.000012</td>
      <td>0.003397</td>
      <td>-2.499020</td>
      <td>0.811181</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-14</th>
      <td>2652.010010</td>
      <td>-0.004079</td>
      <td>0.001332</td>
      <td>0.004279</td>
      <td>1.663961e-05</td>
      <td>0.000008</td>
      <td>0.002893</td>
      <td>-2.128470</td>
      <td>0.744658</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-15</th>
      <td>2675.810059</td>
      <td>0.008934</td>
      <td>0.002021</td>
      <td>0.004283</td>
      <td>7.982196e-05</td>
      <td>0.000011</td>
      <td>0.003269</td>
      <td>-2.404855</td>
      <td>0.847653</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-18</th>
      <td>2690.159912</td>
      <td>0.005348</td>
      <td>0.001887</td>
      <td>0.004122</td>
      <td>2.860622e-05</td>
      <td>0.000030</td>
      <td>0.005481</td>
      <td>-4.032327</td>
      <td>1.573257</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-12-19</th>
      <td>2681.469971</td>
      <td>-0.003235</td>
      <td>0.001858</td>
      <td>0.004157</td>
      <td>1.046845e-05</td>
      <td>0.000030</td>
      <td>0.005444</td>
      <td>-4.005215</td>
      <td>2.356995</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1218 rows × 11 columns</p>
</div>




```python
BackTest(VaR_EWMA(), "Backtesting Value at Risk with EWMA", "Forward_Return", "VaR_99_10D_EWMA", "Breaches_EWMA", "Breaches_EWMA_Conc")
```

    Percentage of Breaches:  2.955665024630542
    


<img src="/img/Var_Problem_66_1.png" alt="" />       

    



```python

```
