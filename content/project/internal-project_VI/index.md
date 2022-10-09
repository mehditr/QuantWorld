---
title: Volatility with GARCH
author: ''
date: '2022-10-09'
slug: volatility-with-garch
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2022-10-09T22:13:37+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


This method is popularly known as **Generalized
ARCH** or **GARCH** model.  
  

\$\$\\sigma^2_n = \\omega + \\sum\_{i=1}^p \\alpha_i u^2\_{n-i} +
\\sum\_{i=1}^q \\beta_i \\sigma^2\_{n-i} \$\$

where, \$p\$ and \$q\$ are lag length.

**GARCH(1,1)** is then represented as,

\$\$ \\sigma^2_n = \\omega + \\alpha u^2\_{n-1} + \\beta
\\sigma^2\_{n-1} \$\$

where, \$\\alpha + \\beta \< 1\$ and \$\\gamma + \\alpha + \\beta = 1\$
as weight applied to long term variance cannot be negative.

where, \$\\frac {\\omega} {(1-\\alpha-\\beta)}\$ is the long-run
variance.

The GARCH model is a way of specifying the dependence of the time
varying nature of volatility. The model incorporates changes in the
fluctuations in volatility and tracks the persistence of volatility as
it fluctuates around its long-term average and are exponentially
weighted.

To model GARCH or the conditional volatility, we need to derive
\$\\omega\$, \$\\alpha\$, \$\\beta\$ by maximizing the likelihood
function.
  
# Required Libraries


```python
# Data manipulation
import pandas as pd
import numpy as np
import yfinance as yf

from scipy.stats import norm
from scipy.optimize import minimize

# Import matplotlib for visualization
import matplotlib
import matplotlib.pyplot as plt

# Plot settings
plt.style.use('dark_background')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['grid.color'] = 'black'

import warnings
warnings.filterwarnings('ignore')
```

# Data: SP500 


```python
df=yf.download('^GSPC',progress=False)
```


```python
df.to_excel(r'C:\Users\Mehdi\Desktop\Data\SP500.xlsx')
```


```python
# Load locally stored data
df = pd.read_excel(r'C:\Users\Mehdi\Desktop\Data\SP500.xlsx', parse_dates=True, index_col=0)['Adj Close']
df = df['2009':'2020']

# Check first and last 5 values 

df
```




    Date
    2009-01-02     931.799988
    2009-01-05     927.450012
    2009-01-06     934.700012
    2009-01-07     906.650024
    2009-01-08     909.729980
                     ...     
    2020-12-24    3703.060059
    2020-12-28    3735.360107
    2020-12-29    3727.040039
    2020-12-30    3732.040039
    2020-12-31    3756.070068
    Name: Adj Close, Length: 3021, dtype: float64




```python
# Visualize FTSE 100 Index Price
plt.plot(df, color='orange')
plt.title('SPX Index');
```


    
![png](GARCH_files/GARCH_7_0.png)
    


# Calculate Log Returns


```python
#Calculate daily returns
# returns = df.pct_change().fillna(0)
returns = np.log(df).diff().fillna(0)
returns
```




    Date
    2009-01-02    0.000000
    2009-01-05   -0.004679
    2009-01-06    0.007787
    2009-01-07   -0.030469
    2009-01-08    0.003391
                    ...   
    2020-12-24    0.003530
    2020-12-28    0.008685
    2020-12-29   -0.002230
    2020-12-30    0.001341
    2020-12-31    0.006418
    Name: Adj Close, Length: 3021, dtype: float64




```python
# Visualize FTSE 100 Index daily returns
plt.plot(returns, color='orange')
plt.title('SPX Index Returns')
```




    Text(0.5, 1.0, 'SPX Index Returns')




    
![png](GARCH_files/GARCH_10_1.png)
    


# Numerical Optimization

I will use Numerical optimization to maximize the likelihood estimation


```python
# GARCH(1,1) function
def garch(omega, alpha, beta, ret):
    
    var = []
    for i in range(len(ret)):
        if i==0:
            var.append(omega/np.abs(1-alpha-beta))
        else:
            var.append(omega + alpha * ret[i-1]**2 + beta * var[i-1])
            
    return np.array(var)
```


```python
garch(np.var(returns),0.1,0.8,returns)[:3]
```




    array([0.00136366, 0.00122729, 0.00112039])



### Maximum Likelihood Estimation<a href="#Maximum-Likelihood-Estimation" class="anchor-link">¶</a>

When using MLE, we first assume a
distribution (ie., a parametric model) and then try to determine the
model parameters. To estimate GARCH(1,1) parameters, we assume
distribution of returns conditional on variance are normally
distributed.

We maximize,

\$\$\\sum\_{i=1}^n log \\Big\[\\frac{1}{\\sqrt{2 \\pi \\sigma_i^2}}
e^{-\\frac{(u_i - \\bar{u})^2}{2 \\sigma_i^2}} \\Big\]\$\$

to derive \$\\omega\$, \$\\alpha\$ and \$\\beta\$.


```python
# Log likelihood function
def MLE(params, ret):
    
    omega= params[0]; alpha = params[1]; beta = params[2]
    
    variance = garch(omega, alpha, beta, ret) # GARCH(1,1) function

    llh = []
    for i in range(len(ret)):
        llh.append(np.log(norm.pdf(ret[i], 0, np.sqrt(variance[i]))))
    
    return -np.sum(np.array(llh))
```


```python
MLE((np.var(returns), 0.1, 0.8), returns)
```




    -7881.868784195199



# Optimization

To optimize the GARCH parameters, we will use the minimize function from scipy optimization module. The objective function here is a function returning maximum log likelihood and the target variables are GARCH parameters.

Further, we use the Nelder–Mead method also known as downhill simplex method which is a commonly applied to numerical method to find the minimum or maximum of an objective function in a multidimensional space.


```python
# Specify optimization input
param = ['omega', 'alpha', 'beta']
initial_values = ((np.var(returns), 0.1, 0.8))
```


```python
res = minimize(MLE, initial_values, args = returns,  method='Nelder-Mead', options={'disp':False})
res
```




     final_simplex: (array([[3.02398346e-06, 1.66716946e-01, 8.15417329e-01],
           [3.02271806e-06, 1.66787920e-01, 8.15398228e-01],
           [3.02377952e-06, 1.66744261e-01, 8.15432017e-01],
           [3.02149237e-06, 1.66710993e-01, 8.15469573e-01]]), array([-9960.80806965, -9960.80806879, -9960.8080682 , -9960.80806387]))
               fun: -9960.808069647774
           message: 'Optimization terminated successfully.'
              nfev: 244
               nit: 136
            status: 0
           success: True
                 x: array([3.02398346e-06, 1.66716946e-01, 8.15417329e-01])




```python
# GARCH parameters
dict(zip(param,np.around(res['x']*100,4)))
```




    {'omega': 0.0003, 'alpha': 16.6717, 'beta': 81.5417}




```python
# Parameters
omega = res['x'][0]; alpha = res['x'][1]; beta = res['x'][2]

# Variance
var = garch(res['x'][0],res['x'][1],res['x'][2],returns)

# Annualised conditional volatility
ann_vol = np.sqrt(var*252) * 100
ann_vol
```




    array([20.6528341 , 18.85280309, 17.51118855, ..., 10.54680756,
           10.02060387,  9.50019167])




```python
# Visualise GARCH volatility and VIX
plt.title('Annualized Volatility')
plt.plot(returns.index, ann_vol, color='orange', label='GARCH')
plt.legend(loc=2);
```


    
![png](GARCH_files/GARCH_24_0.png)
    


# N-day Forecast

Extending the GARCH(1,1) model to forecast future volatility, we can
derive the n-days ahead forecast using the following equation.

\$\$ E\[\\sigma^2\_{n+k}\] = \\overline{\\sigma}\\space{^2} +
(\\alpha+\\beta)^k \* (\\sigma^2_n - \\overline{\\sigma}\\space{^2})
\$\$

where, \$\\overline{\\sigma}\\space{^2}\$ is the long run variance and
\$\\alpha\$ and \$\\beta\$ are GARCH parameters.

We know that volatility has the tendency to revert to its long run
range. And, \$\\alpha + \\beta \< 1\$ in GARCH(1,1) and hence when k
gets larger, the second term gets smaller and the forecast tends towards
the long term variance.



```python
# long run variance
np.sqrt(252*omega/(1-alpha-beta))*100
```




    20.652834103193655




```python
# Calculate N-day forecast
longrun_variance = omega/(1-alpha-beta)
 
fvar = []
for i in range(1,732):    
    fvar.append(longrun_variance + (alpha+beta)**i * (var[-1] - longrun_variance))

var = np.array(fvar)
```


```python
# Verify first 10 values
var[:10]
```




    array([3.81990609e-05, 4.05405904e-05, 4.28402868e-05, 4.50988975e-05,
           4.73171565e-05, 4.94957847e-05, 5.16354900e-05, 5.37369681e-05,
           5.58009016e-05, 5.78279615e-05])




```python
# Plot volatility forecast over different time horizon
plt.axhline(y=np.sqrt(longrun_variance*252)*100, color='yellow')
plt.plot(np.sqrt(var[:300]*252)*100, color='red')

plt.xlabel('Horizon (in days)')
plt.ylabel('Volatility (%)')

plt.annotate('GARCH Forecast', xy=(200,19), color='red')
plt.annotate('Longrun Volatility =' + str(np.around(np.sqrt(longrun_variance*252)*100,2)) + '%', 
             xy=(200,19.50), color='yellow')

plt.title('Volatility Forecast : N-days Ahead')
plt.grid(axis='x')
```


    
![png](GARCH_files/GARCH_30_0.png)
    

