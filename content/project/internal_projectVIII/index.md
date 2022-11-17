---
title: Volatility Prediction
author: ''
date: '2022-11-17'
slug: volatility-prediction
categories: []
tags: []
subtitle: ''
summary: 'Volatiltiy Prediction - Old Fashioned way or Modern Techniques'
authors: []
lastmod: '2022-11-17T13:35:30+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---


# Volatiltiy Prediction - Old Fashioned way or Modern Techniques

In finance volatility is the backbone. Many trades, investors and investment department are following how to evaluate the volatiltiy prediction regarding the activity in various financial fields. Todays, financial markets are more integrated than before and we can see the haevy shadow of uncertainty over all. That makes the uncertainity important notion.

Volatility is frequently used as a proxy of risk in risk management and asset pricing topics as we can see everyday recommendation in financial articles and official papers such as Basel Accord and Solvency. Therefore it is notable for institution to have a reliable model to predict the volatility. For many years quantitative analysts were using , let say, the traditional or, in other words, robust statistical method to predict volatiltiy such ARCH, GARCH and the extended versions. Although many investors are still using these methods as benchmark or somehow for their worste case scenarios, machine learning approaches in financial fields are coming more popular than before.
Using machine learning techniques help analysts to have more percise estimation and concequently they are more informed, how much of their assets must be involved in the different investment markets. 

For this reason I am motivated to see which approach is better by comparison between three traditional model, ARCH, GARCH and GJR-GARCH and two machine learning techniques which are appropriate in stock, indicies or ETF analysis such as SVR and XGBoosting. 

will show you the mathematics behind the ARCH and GARCH and we will check the predictions in each step, based on these methods. The mathematical part can help us to understand what happens exactly behind the models and why they are different from each other.

Then we are able to answer this question:
Is there a significant difference between traditional fashioned way and ML techniques? 
I have to mention that this experiment is just a glimpse of many experiments, however in realy market we have to check both way with more techniques.

# Required Libraries


```python
# Data manipulation
import pandas as pd
import numpy as np
import yfinance as yf
# time 
import datetime
import time
# data modelling
from scipy.stats import norm
from scipy.optimize import minimize
import scipy.optimize as opt
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
# Import matplotlib for visualization
import matplotlib
import matplotlib.pyplot as plt

# Plot settings
plt.style.use('bmh')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['grid.color'] = 'black'

import warnings
warnings.filterwarnings('ignore')
```


```python
stocks='^GSPC'

start= datetime.datetime(2009,1,1)
end=datetime.datetime(2021,8,1)
#(2021,8,1)
#2020,12,31
sp500= yf.download(stocks,start=start,end=end, interval='1d')
```

    [*********************100%***********************]  1 of 1 completed
    


```python
ret = 100*(sp500.pct_change()[1:]['Adj Close'])
realized_vol = ret.rolling(5).std()
```


```python
plt.figure(figsize=(16,5))
plt.plot(realized_vol.index,realized_vol)

```




    [<matplotlib.lines.Line2D at 0xace2541400>]




 <img src="/img/GARCH_6_1.png" alt="" />      

    



```python
n=252
```


```python
split_data = ret.iloc[-n:].index
```


```python
sigma = ret.var()
K = ret.kurtosis()
alpha = (-3.0*sigma + np.sqrt(9.0* sigma**2 -12.0 *(-3.0* sigma - K)*K))/(6*K)
omega = (1-alpha) * sigma
params = [alpha,omega]
params
```




    [0.6124410497162043, 0.514174168530969]



# Autoregressive Conditional Heteroskedasticity - ARCH

ARCH, first proposedby Engel and then it has been developed by other statisticians and mathematicians. Suppose $I_{t}=\{R_{t},R_{t-1},\ldots,R_{1}\}$ is an information set at time t and $R_{t}$ is a daily return of an asset.

The statistical properties of ARCH is as follows:

   - $ E(\epsilon_{t}|I_{t})=0$
   
   - $\mathrm{var}(R_{t}|I_{t})=E(\epsilon_{t}^{2}|I_{t})=\sigma_{t}^{2}$
   
   - $\mathrm{var}(R_{t})=E(\epsilon_{t}^{2})=E(\sigma_{t}^{2})=\omega/(1-\alpha_{1})$
   
   - $\{R_{t}\}$  is an uncorrelated process that means : $ \mathrm{cov}(R_{t},R_{t-k})=E(\epsilon_{t}\epsilon_{t-j})=0 $
   
   - The distribution of $R_{t}$ conditional on $I_{t-1}$ is normal with mean $mu$ and variance $\sigma_{t}^{2}$
   
   - The unconditional (marginal) distribution of $R_{t}$ is not normal and $\mathrm{kurt}(R_{t})\geq3$
   
   - $\{R_{t}^{2}\}$ and $\{\epsilon_{t}^{2}\}$ have a covariance stationary AR(1) model representation. The persistence of the autocorrelations is measured by $\alpha_{1}$.

These properties of the ARCH(1) model match many of the stylized facts of daily asset returns

The ARCH formula is :
\$\$ \\sigma^2_n = \\omega + \\sum\_{i=1}^p \\alpha_i u^2\_{n-i} \$\$


```python
ret_vector = ret.values
```


```python
def arch_MLE(params,ret):
    omega = abs(params[1])
    alpha = abs(params[0])
    
    T = len(ret_vector)
    sigma_2 = np.zeros(T)
    MLE_arch = 0
    sigma_2[0]=np.var(ret_vector)
    
    for t in range(1,T):
        sigma_2[t] = omega + alpha * (ret_vector[t-1])**2
    MLE_arch = np.sum(0.5 * (np.log(sigma_2)+ret_vector**2 / sigma_2))
    return MLE_arch
    

```


```python
arch_MLE(params,ret_vector)
```




    1815.8975757563007




```python
def optim_params(x0,ret_vector):
    results = opt.minimize(arch_MLE, x0=x0 , args = (ret_vector), method='Nelder-Mead',options ={'maxiter':5000})
    
    params_2 = results.x
    print('\nResults of Nelder-Mead minimization\n{}\n{}'.format(''.join(['-'] * 28), results))
    print('\nResulting params = {}'.format(params_2))
    return params_2
```


```python
 drived_params = optim_params(params,ret_vector)
```

    
    Results of Nelder-Mead minimization
    ----------------------------
     final_simplex: (array([[0.40232949, 0.80637059],
           [0.40234986, 0.80643784],
           [0.40242228, 0.80638205]]), array([1728.14722176, 1728.14722211, 1728.14722275]))
               fun: 1728.1472217577239
           message: 'Optimization terminated successfully.'
              nfev: 65
               nit: 33
            status: 0
           success: True
                 x: array([0.40232949, 0.80637059])
    
    Resulting params = [0.40232949 0.80637059]
    


```python
def arch_func(ret):
    omega = drived_params[1]
    alpha = drived_params[0]
    
    T = len(ret)
    sigma_2_arch = np.zeros(T+1)
    sigma_2_arch[0] = np.var(ret)
    
    for t in range(1,T):
        sigma_2_arch[t] = omega + alpha * (ret[t-1])**2
    return sigma_2_arch
    
    
```


```python
sigma_2_arch = arch_func(ret)
```


```python
from arch import arch_model
arch = arch_model(ret, mean='zero', vol='ARCH' , p=1).fit(disp='off')
print(arch.summary)
```

    <bound method ARCHModelResult.summary of                         Zero Mean - ARCH Model Results                        
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                       ARCH   Log-Likelihood:               -4636.95
    Distribution:                  Normal   AIC:                           9277.89
    Method:            Maximum Likelihood   BIC:                           9290.01
                                            No. Observations:                 3165
    Date:                Thu, Nov 17 2022   Df Residuals:                     3165
    Time:                        13:01:50   Df Model:                            0
                                Volatility Model                            
    ========================================================================
                     coef    std err          t      P>|t|  95.0% Conf. Int.
    ------------------------------------------------------------------------
    omega          0.8067  5.432e-02     14.851  6.847e-50 [  0.700,  0.913]
    alpha[1]       0.4012  7.121e-02      5.634  1.757e-08 [  0.262,  0.541]
    ========================================================================
    
    Covariance estimator: robust
    ARCHModelResult, id: 0xace290c550>
    


```python
ARCH_BIC=[]

for p in range(1,5):
    arch = arch_model( ret, mean = 'zero', vol = 'ARCH', p = p).fit(disp='off')
    ARCH_BIC.append(arch.bic)
    if arch.bic == np.min(ARCH_BIC):
        best_param = p
arch = arch = arch_model( ret, mean = 'zero', vol = 'ARCH', p = best_param).fit(disp='off')
print(arch.summary)
```

    <bound method ARCHModelResult.summary of                         Zero Mean - ARCH Model Results                        
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                       ARCH   Log-Likelihood:               -4203.84
    Distribution:                  Normal   AIC:                           8417.67
    Method:            Maximum Likelihood   BIC:                           8447.97
                                            No. Observations:                 3165
    Date:                Thu, Nov 17 2022   Df Residuals:                     3165
    Time:                        13:01:50   Df Model:                            0
                                 Volatility Model                             
    ==========================================================================
                     coef    std err          t      P>|t|    95.0% Conf. Int.
    --------------------------------------------------------------------------
    omega          0.2852  2.584e-02     11.041  2.432e-28   [  0.235,  0.336]
    alpha[1]       0.1397  3.147e-02      4.440  8.980e-06 [7.806e-02,  0.201]
    alpha[2]       0.2425  3.390e-02      7.154  8.435e-13   [  0.176,  0.309]
    alpha[3]       0.2328  3.505e-02      6.643  3.080e-11   [  0.164,  0.302]
    alpha[4]       0.1997  3.748e-02      5.328  9.952e-08   [  0.126,  0.273]
    ==========================================================================
    
    Covariance estimator: robust
    ARCHModelResult, id: 0xace28e5e80>
    


```python
pred =arch.forecast(start=split_data[0])
pred_arch = pred
```


```python
plt.figure(figsize=(16, 6))
plt.plot(realized_vol / 100, label='Realized Volatility')
plt.plot(pred_arch.variance.iloc[-len(split_data):]/100,label='Volatility Prediction - ARCH')
plt.title('Volatility Prediction with ARCH')
plt.legend()
```




    <matplotlib.legend.Legend at 0xace28e56a0>




    
![png](GARCH_files/GARCH_23_1.png)
    




## GARCH<a href="#GARCH" class="anchor-link">¶</a>

This method is popularly known as **Generalized
ARCH** or **GARCH** model.  
  

\$\$ \\sigma^2_n = \\omega + \\sum\_{i=1}^p \\alpha_i u^2\_{n-i} +
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

# Numerical Optimization


```python
ret = ret/100
```

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
garch(np.var(ret),0.1,0.8,ret)[:3]
```




    array([0.00132628, 0.00119583, 0.0010954 ])



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
def likelihood(parameters, ret):
    
    omega= parameters[0]; alpha = parameters[1]; beta = parameters[2]
    
    variance = garch(omega, alpha, beta, ret) # GARCH(1,1) function

    llh = []
    for i in range(len(ret)):
        llh.append(np.log(norm.pdf(ret[i], 0, np.sqrt(variance[i]))))
    
    return -np.sum(np.array(llh))
```


```python
likelihood((np.var(ret), 0.1, 0.8), ret)
```




    -8299.98799385085



# Optimization

To optimize the GARCH parameters, we will use the minimize function from scipy optimization module. The objective function here is a function returning maximum log likelihood and the target variables are GARCH parameters.

Further, we use the Nelder–Mead method also known as downhill simplex method which is a commonly applied to numerical method to find the minimum or maximum of an objective function in a multidimensional space.


```python
# Specify optimization input
param = ['omega', 'alpha', 'beta']
initial_values = ((np.var(ret), 0.1, 0.8))
```


```python
res = minimize(likelihood, initial_values, args = ret,  method='Nelder-Mead', options={'disp':False})
res
```




     final_simplex: (array([[3.15726982e-06, 1.66123864e-01, 8.13434579e-01],
           [3.15817026e-06, 1.66163035e-01, 8.13412947e-01],
           [3.15645915e-06, 1.66045708e-01, 8.13498081e-01],
           [3.15407348e-06, 1.66092987e-01, 8.13528261e-01]]), array([-10447.92051592, -10447.92051307, -10447.92050653, -10447.92050374]))
               fun: -10447.920515918144
           message: 'Optimization terminated successfully.'
              nfev: 237
               nit: 131
            status: 0
           success: True
                 x: array([3.15726982e-06, 1.66123864e-01, 8.13434579e-01])




```python
# GARCH parameters
dict(zip(param,np.around(res['x'],8)))
```




    {'omega': 3.16e-06, 'alpha': 0.16612386, 'beta': 0.81343458}




```python
# Parameters
omega = res['x'][0]; alpha = res['x'][1]; beta = res['x'][2]

# Variance
var = garch(res['x'][0],res['x'][1],res['x'][2],ret)

# Annualised conditional volatility
ann_vol = np.sqrt(var*252) * 100
ann_vol
```




    array([19.72873025, 18.26710399, 17.46340537, ..., 13.32892214,
           12.34851269, 11.80677087])



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




    19.72873025348311




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




    array([5.73438923e-05, 5.93289636e-05, 6.12734571e-05, 6.31782020e-05,
           6.50440110e-05, 6.68716800e-05, 6.86619886e-05, 7.04157005e-05,
           7.21335637e-05, 7.38163112e-05])




```python
# Plot volatility forecast over different time horizon
plt.axhline(y=np.sqrt(longrun_variance*252)*100, color='yellow')
plt.plot(np.sqrt(var[:300]*252)*100, color='red')

plt.xlabel('Horizon (in days)')
plt.ylabel('Volatility (%)')

plt.annotate('GARCH Forecast', xy=(200,19), color='red')
plt.annotate('Longrun Volatility =' + str(np.around(np.sqrt(longrun_variance*252)*100,2)) + '%', 
             xy=(200,19.50), color='blue')

plt.title('Volatility Forecast : N-days Ahead')
plt.grid(axis='x')
```


    
![png](GARCH_files/GARCH_44_0.png)
    



```python
g1 = arch_model(ret, mean='zero', vol='GARCH', p=1, o=0, q=1,dist='Normal')
```


```python
model = g1.fit()
```

    Iteration:      1,   Func. Count:      4,   Neg. LLF: -10436.555895862148
    Optimization terminated successfully    (Exit mode 0)
                Current function value: -10436.555883468754
                Iterations: 5
                Function evaluations: 4
                Gradient evaluations: 1
    


```python
print(model)
```

                           Zero Mean - GARCH Model Results                        
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:                10436.6
    Distribution:                  Normal   AIC:                          -20867.1
    Method:            Maximum Likelihood   BIC:                          -20848.9
                                            No. Observations:                 3165
    Date:                Thu, Nov 17 2022   Df Residuals:                     3165
    Time:                        13:03:28   Df Model:                            0
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega      2.6588e-06  6.301e-10   4219.300      0.000 [2.658e-06,2.660e-06]
    alpha[1]       0.2000  3.035e-02      6.590  4.393e-11     [  0.141,  0.259]
    beta[1]        0.7800  2.351e-02     33.181 2.000e-241     [  0.734,  0.826]
    ============================================================================
    
    Covariance estimator: robust
    


```python
GARCH_BIC = []
ret = ret*100
for p in range(1, 5):
    for q in range(1, 5):
        GARCH = arch_model(ret, mean='zero',vol='GARCH', p=p, o=0, q=q).fit(disp='off')
        GARCH_BIC.append(GARCH.bic)
        if GARCH.bic == np.min(GARCH_BIC):
            best_param = p, q
GARCH = arch_model(ret, mean='zero', vol='GARCH',p=best_param[0], o=0, q=best_param[1]).fit(disp='off')
print(GARCH.summary())
pred_GARCH = GARCH.forecast(start=split_data[0])
prediction_GARCH = pred_GARCH
```

                           Zero Mean - GARCH Model Results                        
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                      GARCH   Log-Likelihood:               -4125.10
    Distribution:                  Normal   AIC:                           8256.21
    Method:            Maximum Likelihood   BIC:                           8274.39
                                            No. Observations:                 3165
    Date:                Thu, Nov 17 2022   Df Residuals:                     3165
    Time:                        13:03:30   Df Model:                            0
                                  Volatility Model                              
    ============================================================================
                     coef    std err          t      P>|t|      95.0% Conf. Int.
    ----------------------------------------------------------------------------
    omega          0.0322  7.085e-03      4.542  5.580e-06 [1.829e-02,4.606e-02]
    alpha[1]       0.1562  2.003e-02      7.797  6.348e-15     [  0.117,  0.195]
    beta[1]        0.8170  1.952e-02     41.863      0.000     [  0.779,  0.855]
    ============================================================================
    
    Covariance estimator: robust
    


```python
RMSE_GARCH = np.sqrt(mse(realized_vol[-n:] / 100,np.sqrt(prediction_GARCH.variance.iloc[-len(split_data):] / 100)))
print('The RMSE value of GARCH model is {:.4f}'.format(RMSE_GARCH))
plt.figure(figsize=(16,6))
plt.plot(realized_vol / 100, label='Realized Volatility')
plt.plot(prediction_GARCH.variance.iloc[-len(split_data):] / 100, 
         label='Volatility Prediction-GARCH')
plt.title('Volatility Prediction with GARCH', fontsize=12)
plt.legend()
plt.show()
```

    The RMSE value of GARCH model is 0.0885
    


    
![png](GARCH_files/GARCH_49_1.png)
    


One of the GARCH extended model that perfomrs well for asymetric returns is called GJR-GARCH. Since the asymmetric returns happen, the fatter tail exists in the distribution of losses, rather than the distribution of gains. In GJR-GARCH there is another parameter, called $\gamma$, that controls the asymmetric part. When the $\gamma = 0$, the formula is simply the GARCH and that means the effect of asymmetry is the same as past. $\gamma > 0$ tells us that past negative shock prevails its positive shock and finally  $\gamma <0$ the positive shock prevails over negative one.

I am not going to manually calculate GJR-GARCH as I show it in a different post, but I will show how the effect of GJR-GARCH on RMSE and we will see how it behaves in volatility prediction




```python
GJR_GARCH_BIC = []

for p in range(1, 5):
    for q in range(1, 5):
        GJR_GARCH = arch_model(ret, mean='zero', p=p, o=1, q=q).fit(disp='off')
        GJR_GARCH_BIC.append(GJR_GARCH.bic)
        if GJR_GARCH.bic == np.min(GJR_GARCH_BIC):
            best_param = p, q
gjrgarch = arch_model(ret,mean='zero', p=best_param[0], o=1,q=best_param[1]).fit(disp='off')
print(GJR_GARCH.summary())
pred_GJR_GARCH = GJR_GARCH.forecast(start=split_data[0])
prediction_GJR_GARCH = pred_GJR_GARCH
```

                         Zero Mean - GJR-GARCH Model Results                      
    ==============================================================================
    Dep. Variable:              Adj Close   R-squared:                       0.000
    Mean Model:                 Zero Mean   Adj. R-squared:                  0.000
    Vol Model:                  GJR-GARCH   Log-Likelihood:               -4054.85
    Distribution:                  Normal   AIC:                           8129.71
    Method:            Maximum Likelihood   BIC:                           8190.31
                                            No. Observations:                 3165
    Date:                Thu, Nov 17 2022   Df Residuals:                     3165
    Time:                        13:03:43   Df Model:                            0
                                   Volatility Model                              
    =============================================================================
                     coef    std err          t      P>|t|       95.0% Conf. Int.
    -----------------------------------------------------------------------------
    omega          0.0454  1.097e-02      4.133  3.573e-05  [2.385e-02,6.686e-02]
    alpha[1]   2.2691e-03  4.715e-02  4.813e-02      0.962 [-9.014e-02,9.468e-02]
    alpha[2]       0.0000  4.907e-02      0.000      1.000 [-9.618e-02,9.618e-02]
    alpha[3]       0.0000  4.631e-02      0.000      1.000 [-9.076e-02,9.076e-02]
    alpha[4]       0.0502  4.048e-02      1.241      0.215   [-2.911e-02,  0.130]
    gamma[1]       0.2941  6.046e-02      4.865  1.146e-06      [  0.176,  0.413]
    beta[1]        0.7542      0.321      2.351  1.873e-02      [  0.125,  1.383]
    beta[2]        0.0000      0.276      0.000      1.000      [ -0.541,  0.541]
    beta[3]        0.0000      0.367      0.000      1.000      [ -0.719,  0.719]
    beta[4]        0.0168      0.252  6.676e-02      0.947      [ -0.477,  0.511]
    =============================================================================
    
    Covariance estimator: robust
    


```python
GJR_GARCH_RMSE = np.sqrt(mse(realized_vol[-n:] / 100,
                             np.sqrt(prediction_GJR_GARCH.variance.iloc[-len(split_data):]/ 100)))
print('The RMSE value of GJR-GARCH models is {:.4f}'.format(GJR_GARCH_RMSE))

plt.figure(figsize=(16, 6))
plt.plot(realized_vol / 100, label='Realized Volatility')
plt.plot(prediction_GJR_GARCH.variance.iloc[-len(split_data):] / 100, 
         label='Volatility Prediction-GJR-GARCH')
plt.title('Volatility Prediction with GJR-GARCH', fontsize=12)
plt.legend()
plt.show()
```

    The RMSE value of GJR-GARCH models is 0.0887
    


    
![png](GARCH_files/GARCH_52_1.png)
    


# Support Vector Machine & Gradient Boosting Approaches 

Support vector machine is a popular supervised machine learning approach and can be applied in both classification and regression problem. The general idea behind support vector machine is, choose the best line, or in linear algebra terminology a hyperplane, that maximizes the distance among points that are closest to the line or,the 'hyperplane' but actually they belong to anothe class.
we call SVC from SVM for classification purposes and SVR for regression goals. In the regression that I will apply it, the aim is to find that minimizes the error and maximizes the margin.There are two types of margin, hard and soft and the hard margine is narrower than soft margine and it performs as the same as regularization technique. Soft margine allows more misclassified points in the model.


```python
realized_vol = ret.rolling(5).std()
realized_vol = pd.DataFrame(realized_vol)
realized_vol.reset_index(drop=True, inplace=True)
```


```python
returns_svm = ret ** 2
returns_svm = returns_svm.reset_index()
del returns_svm['Date']
```


```python
X = pd.concat([realized_vol,returns_svm],axis=1,ignore_index=True)
X = X[4:].copy()
X = X.reset_index()
X.drop('index',axis=1,inplace=True)
```


```python
realized_vol = realized_vol.dropna().reset_index()
```


```python
realized_vol.drop('index',axis=1,inplace=True)
```


```python
def svm_report(X,y,kernel='linear',n=252):
    C= sp_rand(); gamma=sp_rand()
    svr=SVR(kernel=kernel,C= sp_rand(), gamma=sp_rand())
    para_grid={'gamma':gamma,'C':C,'epsilon':sp_rand()}
    clf = RandomizedSearchCV(svr,para_grid)
    clf.fit(X.iloc[:-n].values,y.iloc[1:-(n-1)].values.reshape(-1,))
    predict_svr = clf.predict(X.iloc[-n:])
    predict_svr = pd.DataFrame(predict_svr)
    predict_svr.index = ret.iloc[-n:].index
    y.index=ret.iloc[4:].index
    
    rmse_svr = np.sqrt(mse(y.iloc[-n:] /100,predict_svr/100))
    print(' The RMSE value of SVR with {} Kernel is {:.6f}'.format(kernel,rmse_svr))
    
    plt.figure(figsize=(20,10))
    plt.plot(realized_vol /100, label="Realized Volatility")
    plt.plot(predict_svr /100, label = 'Volatility Prediction - SVR - GARCH')
    plt.title('Volatility Prediction with SVR-GARCH ({})'.format(kernel))
    plt.legend()
```


```python
svm_report(X,realized_vol,kernel='rbf',n=252)
```

     The RMSE value of SVR with rbf Kernel is 0.000879
    


    
![png](GARCH_files/GARCH_61_1.png)
    



```python
svm_report(X,realized_vol,kernel='linear',n=252)
```

     The RMSE value of SVR with linear Kernel is 0.000455
    


    
![png](GARCH_files/GARCH_62_1.png)
    



```python
# Hyper parameter optimization
# Scale and fit the classifier model
xgbcls =  XGBRegressor(verbosity = 0, silent=True, random_state=42)
param_grid = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
              'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
              'min_child_weight': [1, 3, 5, 7],
              'gamma': [0.0, 0.1, 0.2 , 0.3, 0.4]}
rs = RandomizedSearchCV(xgbcls, param_grid, n_iter=100,cv=5, verbose=1)
```


```python
rs.fit(X.iloc[:-n].values,realized_vol.iloc[1:-(n-1)].values.reshape(-1,), verbose=1)
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    




    RandomizedSearchCV(cv=5,
                       estimator=XGBRegressor(base_score=None, booster=None,
                                              callbacks=None,
                                              colsample_bylevel=None,
                                              colsample_bynode=None,
                                              colsample_bytree=None,
                                              early_stopping_rounds=None,
                                              enable_categorical=False,
                                              eval_metric=None, gamma=None,
                                              gpu_id=None, grow_policy=None,
                                              importance_type=None,
                                              interaction_constraints=None,
                                              learning_rate=None, max_bin=None,
                                              m...
                                              min_child_weight=None, missing=nan,
                                              monotone_constraints=None,
                                              n_estimators=100, n_jobs=None,
                                              num_parallel_tree=None,
                                              predictor=None, random_state=42,
                                              reg_alpha=None, reg_lambda=None, ...),
                       n_iter=100,
                       param_distributions={'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                                            'learning_rate': [0.05, 0.1, 0.15, 0.2,
                                                              0.25, 0.3],
                                            'max_depth': [3, 4, 5, 6, 8, 10, 12,
                                                          15],
                                            'min_child_weight': [1, 3, 5, 7]},
                       verbose=1)




```python
predict_xgboost = rs.predict(X.iloc[-n:])
```


```python
predict_xgboost = pd.DataFrame(predict_xgboost)
predict_xgboost.index = ret.iloc[-n:].index
```


```python
rmse_xgboost = np.sqrt(mse(realized_vol.iloc[-n:] / 100,
                       predict_xgboost / 100))
print('The RMSE value of SVR with Linear Kernel is {:.6f}'
      .format(rmse_xgboost))
```

    The RMSE value of SVR with Linear Kernel is 0.000623
    


```python
realized_vol.index = ret.iloc[4:].index
```


```python
plt.figure(figsize=(20, 10))
plt.plot(realized_vol / 100, label='Realized Volatility')
plt.plot(predict_xgboost / 100, label='Volatility Prediction-SVR-GARCH')
plt.title('Volatility Prediction with SVR-GARCH (Linear)', fontsize=12)
print('The RMSE value of XGBoost with Linear Kernel is {:.6f}'.format(rmse_xgboost))
plt.legend()
plt.show()
```

    The RMSE value of XGBoost with Linear Kernel is 0.000623
    


    
![png](GARCH_files/GARCH_69_1.png)
    


# Conclusion

In summary, as we have seen at the SVR and XGBoosting outperforme rather than the ARCH-GARCH models in respect with the RMSEs. The statistical models are more overstimated and might be good in respect of Basel Accord approaches, but in internal reports one might use ML techniques. However, this is what we have seen according to the ticker ***S&P500*** in the given time. I am almost pretty sure for other tickers or different industries such gold or Gold ETF the result might be changed and GJR-GARCH is probably pioneer there. We must pay attention that ***monitoring*** is a crucial step in risk managemnt and for getting a reliable and stable model, we should monitor our result in different aspects.


```python

```
