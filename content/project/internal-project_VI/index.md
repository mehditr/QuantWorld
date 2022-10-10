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


## Libraries and Modules

We need the following Python libraries and modules:

```{python}
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import scipy.optimize as sco
import scipy.interpolate as sci
```

---

## Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For our sample, we select 12 different assets for the analysis with exposure to all of the GICS sectors—-- energy, materials, industrials, utilities, healthcare, financials, consumer discretionary, consumer staples, information technology, communication services, and real estate. The start date is 2011-09-13 and the end date is by default today's date, which is `r Sys.Date()`.

```{python}
# Create a list of symbols
symbols = [
  "XOM", "SHW", "JPM", "AEP", "UNH", "AMZN", 
  "KO", "BA", "AMT", "DD", "TSN", "SLG"
]
# Instantiate data frame container
asset_data = pd.DataFrame()
# For loop to get data from Yahoo finance
for sym in symbols:
  # Each run of the loop returns a pandas data frame
  asset_data[sym] = web.DataReader(
    name = sym, 
    data_source = 'yahoo',
    start = '2011-09-13'
    # Use [ to extract values as pandas series
    )['Adj Close']
# Set column indices
asset_data.columns = symbols
# Examine the first 5 rows
asset_data.head(n = 5)
```

Next, we find the simple daily returns for each of the 12 assets using the `pct_change()` method, since our data object is a Pandas `DataFrame`. We use simple returns since they have the property of being asset-additive, which is necessary since we need to compute portfolios returns:

```{python}
# Compute daily simple returns
daily_returns = (
  asset_data.pct_change()
            .dropna(
              # Drop the first row since we have NaN's
              # The first date 2011-09-13 does not have a value since it is our cut-off date
              axis = 0,
              how = 'any',
              inplace = False
              )
)
# Examine the last 5 rows
daily_returns.tail(n = 5)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The simple daily returns may be visualized using line charts, density plots, and histograms, which are covered in my other [post on visualizing asset data](https://www.kenwuyang.com/en/post/visualizing-asset-returns/). Even though the visualizations in that post use the `ggplot2` package in R, the `plotnine` package, or any other Python graphics librarires, can be employed to produce them in Python. For now, let us annualize the daily returns over the 10-year period from 2011-9-13 to `r Sys.Date()`. We assume the number of trading days in a year is computed as follows:

\begin{align*}
365.25 \text{(days on average per year)} \times \frac{5}{7} \text{(proportion work days per week)} \\
- 6 \text{(weekday holidays)} - 3\times\frac{5}{7} \text{(fixed date holidays)} = 252.75 \approx 253
\end{align*}

```{python}
daily_returns.mean() * 253
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As can be seen, there are significant differences in the annualized performances between these assets. Amazon dominated the period with an annualized rate of return of $32.2\%$. Exxon Mobil Corporation, on the other hand, is at the bottom of the rankings, recording an annualized rate of return of only $4.2\%$ over the period. The annualized variance-covariance matrix of the returns can be computed using built-in `pandas` method `cov`:

```{python}
daily_returns.cov() * 253
```

The variance-covariance matrix of the returns will be needed to compute the variance of the portfolio returns.

---

## The Optimization Problem

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The portfolio optimization problem, therefore, given a universe of assets and their characteristics, deals with a method to spread the capital between them in a way that *maximizes the return of the portfolio per unit of risk taken*. There is no unique solution for this problem, but a set of solutions, which together define what is called an **efficient frontier**--- the portfolios whose returns cannot be improved without increasing risk, or the portfolios where risk cannot be reduced without reducing returns as well. The Markowitz model for the solution of the portfolio optimization problem has a twin objective of maximizing return and minimizing risk, built on the Mean-Variance framework of asset returns and holding the basic constraints, which reduces to the following:

**Minimize Risk given Levels of Return**

\begin{align*}
\min_{\vec{w}} \hspace{5mm} \sqrt{\vec{w}^{T} \hat{\Sigma} \vec{w}}
\end{align*}

subject to 

\begin{align*}
&\vec{w}^{T} \hat{\mu}=\bar{r}_{P} \\
&\vec{w}^{T} \vec{1} = 1 \hspace{5mm} (\text{Full investment}) \\
&\vec{0} \le \vec{w} \le \vec{1} \hspace{5mm} (\text{Long only})
\end{align*}

**Maximize Return given Levels of Risk**

\begin{align*}
\max _{\vec{w}} \hspace{5mm} \vec{w}^{T} \hat{\mu}
\end{align*}

subject to

\begin{align*}
&\vec{w}^{T} \hat{\Sigma} \vec{w}=\bar{\sigma}_{P} \\
&\vec{w}^{T} \vec{1} = 1 \hspace{5mm} (\text{Full investment}) \\
&\vec{0} \le \vec{w} \le \vec{1} \hspace{5mm} (\text{Long only})
\end{align*}

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In absence of other constraints, the above model is loosely referred to as the "unconstrained" portfolio optimization model. Solving the mathematical model yields a set of optimal weights representing a set of optimal portfolios. The solution set to these two problems is a hyperbola that depicts the efficient frontier in the $\mu-\sigma$ -diagram.

---

## Monte Carlo Simulation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first task is to simulate a random set of portfolios to visualize the risk-return profiles of our given set of assets. To carry out the Monte Carlo simulation, we define two functions that both take as inputs a vector of asset weights and output the expected portfolio return and standard deviation:

**Returns**

```{python}
# Function for computing portfolio return
def portfolio_returns(weights):
    return (np.sum(daily_returns.mean() * weights)) * 253
```

**Standard Deviation**

```{python}
# Function for computing standard deviation of portfolio returns
def portfolio_sd(weights):
    return np.sqrt(np.transpose(weights) @ (daily_returns.cov() * 253) @ weights)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Next, we use a for loop to simulate random vectors of asset weights, computing the expected portfolio return and standard deviation for each permutation of random weights. Again, we ensure that each random weight vector is subject to the long-positions-only and full-investment constraints.

**Monte Carlo Simulation**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The empty containers we instantiate are lists; they are *mutable* and so growing them will not be memory inefficient. 

```{python}
# instantiate empty list containers for returns and sd
list_portfolio_returns = []
list_portfolio_sd = []
# For loop to simulate 5000 random weight vectors (numpy array objects)
for p in range(5000):
  # Return random floats in the half-open interval [0.0, 1.0)
  weights = np.random.random(size = len(symbols)) 
  # Normalize to unity
  # The /= operator divides the array by the sum of the array and rebinds "weights" to the new object
  weights /= np.sum(weights) 
  # Lists are mutable so growing will not be memory inefficient
  list_portfolio_returns.append(portfolio_returns(weights))
  list_portfolio_sd.append(portfolio_sd(weights))
  # Convert list to numpy arrays
  port_returns = np.array(object = list_portfolio_returns)
  port_sd = np.array(object = list_portfolio_sd)
```

Let us examine the simulation results. In particular, the highest and the lowest expected portfolio returns are as follows:

```{python}
# Max expected return
round(max(port_returns), 4)
# Min expected return
round(min(port_returns), 4)
```

On the other hand, the highest and lowest volatility measures are recorded as:

```{python}
# Max sd
round(max(port_sd), 4)
# Min sd
round(min(port_sd), 4)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We may also visualize the expected returns and standard deviations on a $\mu-\sigma$ *trade-off* diagram. For this task, I will leverage R's graphics engine and the `plotly` graphics library. The `reticulate` package in R allows for relatively seamless transition between Python and R. Fortunately, the NumPy arrays created in Python can be accessed as R vector objects; this makes plotting in R using Python objects simple:

```{r}
# Plot the sub-optimal portfolios
plotly::plot_ly(
  x = py$port_sd, y = py$port_returns, color = (py$port_returns / py$port_sd),
  mode = "markers", type = "scattergl", showlegend = FALSE,
  marker = list(size = 5, opacity = 0.7)
) %>%
  plotly::layout(
    title = "Mean-Standard Deviation Diagram",
    yaxis = list(title = "Expected Portfolio Return (Annualized)", tickformat = ".2%"),
    xaxis = list(title = "Portoflio Standard Deviation (Annualized)", tickformat = ".2%")
  ) %>% 
  plotly::colorbar(title = "Sharpe Ratio")
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Each point in the diagram above represents a permutation of expected-return-standard-deviation value pair. The points are color coded such that the magnitudes of the Sharpe ratios, defined as $SR ≡ \frac{\mu_{P} – r_{f}}{\sigma_{P}}$, can be readily observed for each expected-return-standard-deviation pairing. For simplicity, we assume that $r_{f} ≡ 0$. It could be argued that the assumption here is restrictive, so I explored using a different risk-free rate in my previous [post](https://www.kenwuyang.com/en/post/portfolio-optimization-dashboard/). 

---

## The Optimal Portfolios

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Solving the optimization problem defined earlier provides us with a *set of optimal portfolios* given the characteristics of our assets. There are two important portfolios that we may be interested in constructing--- the **minimum variance portfolio** and the **maximal Sharpe ratio portfolio**. In the case of the maximal Sharpe ratio portfolio, the objective function we wish to maximize is our user-defined Sharpe ratio function. The constraint is that all weights sum up to 1. We also specify that the weights are bound between 0 and 1. In order to use the minimization function from the `SciPy` library, we need to transform the maximization problem into one of minimization. In other words, the negative value of the Sharpe ratio is minimized to find the maximum value; the optimal portfolio composition is therefore the array of weights that yields that maximum value of the Sharpe ratio.

```{python}
# User defined Sharpe ratio function
# Negative sign to compute the negative value of Sharpe ratio
def sharpe_fun(weights):
  return - (portfolio_returns(weights) / portfolio_sd(weights))
```

We will use dictionaries inside of a tuple to represent the constraints:

```{python}
# We use an anonymous lambda function
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
```

Next, the bound values for the weights:

```{python}
# This creates 12 tuples of (0, 1), all of which exist within a container tuple
# We essentially create a sequence of (min, max) pairs
bounds = tuple(
  (0, 1) for w in weights
)
```

We also need to supply a starting list of weights, which essentially functions as an initial guess. For our purposes, this will be an equal weight array:

```{python}
# Repeat the list with the value (1 / 12) 12 times, and convert list to array
equal_weights = np.array(
  [1 / len(symbols)] * len(symbols)
)
```

We will use the `scipy.optimize.minimize` function and the Sequential Least Squares Programming (SLSQP) method for the minimization:

```{python}
# Minimization results
max_sharpe_results = sco.minimize(
  # Objective function
  fun = sharpe_fun, 
  # Initial guess, which is the equal weight array
  x0 = equal_weights, 
  method = 'SLSQP',
  bounds = bounds, 
  constraints = constraints
)
```

The class of the optimization results is `scipy.optimize.optimize.OptimizeResult`, which contains many objects. The object of interest to us is the weight composition array, which we employ to construct the maximal Sharpe ratio portfolio:

```{python}
# Extract the weight composition array
max_sharpe_results["x"]
```

### Maximal Sharpe Ratio Portfolio

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Therefore, the expected return, standard deviation, and Sharpe ratio of this portfolio are as follows:

* **Expected return (Maximal Sharpe ratio portfolio)**

```{python}
# Expected return
max_sharpe_port_return = portfolio_returns(max_sharpe_results["x"])
round(max_sharpe_port_return, 4)
```

* **Standard deviation (Maximal Sharpe ratio portfolio)**

```{python}
# Standard deviation
max_sharpe_port_sd = portfolio_sd(max_sharpe_results["x"])
round(max_sharpe_port_sd, 4)
```

* **Sharpe ratio (Maximal Sharpe ratio portfolio)**

```{python}
# Sharpe ratio
max_sharpe_port_sharpe = max_sharpe_port_return / max_sharpe_port_sd
round(max_sharpe_port_sharpe, 4)
```

---

### Minimum Variance Portfolio

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The minimum variance portfolio may be constructed similarly. The objective function, in this case, is the standard deviation function:

```{python}
# Minimize sd
min_sd_results = sco.minimize(
  # Objective function
  fun = portfolio_sd, 
  # Initial guess, which is the equal weight array
  x0 = equal_weights, 
  method = 'SLSQP',
  bounds = bounds, 
  constraints = constraints
)
```

* **Expected return (Minimum variance portfolio)**

```{python}
# Expected return
min_sd_port_return = portfolio_returns(min_sd_results["x"])
round(min_sd_port_return, 4)
```

* **Standard deviation (Minimum variance portfolio)**

```{python}
# Standard deviation
min_sd_port_sd = portfolio_sd(min_sd_results["x"])
round(min_sd_port_sd, 4)
```

* **Sharpe ratio (Minimum variance portfolio)**

```{python}
# Sharpe ratio
min_sd_port_sharpe = min_sd_port_return / min_sd_port_sd
round(min_sd_port_sharpe, 4)
```

---

## Efficient Frontier

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As an investor, one is generally interested in the maximum return given a fixed risk level or the minimum risk given a fixed return expectation. As mentioned in the earlier section, the set of optimal portfolios--- whose expected portfolio returns for a defined level of risk cannot be surpassed by any other portfolio--- depicts the so-called efficient frontier. The Python implementation is to fix a target return level and, for each such level, minimize the volatility value. For the optimization, we essentially "fit" the twin-objective described earlier into an optimization problem that can be solved using quadratic programming. (The objective function is the portfolio standard deviation formula, which is a quadratic function) Therefore, the two linear constraints are the target return (a linear function) and that the weights must sum to 1 (another linear function). We will again use dictionaries inside of a tuple to represent the constraints:

```{python}
# We use anonymous lambda functions
# The argument x will be the weights
constraints = (
  {'type': 'eq', 'fun': lambda x: portfolio_returns(x) - target}, 
  {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The full-investment and long-positions-only specifications will remain unchanged throughout the optimization process, but the value for `target` will be different during each iteration. Since dictionaries are *mutable*, the first constraint will be updated repeatedly during the minimization process. However, because tuples are *immutable*, the references held by the `constraints` tuple will always point to the same objects. This nuance makes the implementation Pythonic. We again constrain the weights such that all weights fall within the interval $[0,1]$:

```{python}
# This creates 12 tuples of (0, 1), all of which exist within a container tuple
# We essentially create a sequence of (min, max) pairs
bounds = tuple(
  (0, 1) for w in weights
)
```

Here's the implementation in Python. We will use the `scipy.optimize.minimize` function again and the Sequential Least Squares Programming (SLSQP) method for the minimization:

```{python}
# Initialize an array of target returns
target = np.linspace(
  start = 0.15, 
  stop = 0.30,
  num = 100
)
# instantiate empty container for the objective values to be minimized
obj_sd = []
# For loop to minimize objective function
for target in target:
  min_result_object = sco.minimize(
    # Objective function
    fun = portfolio_sd, 
    # Initial guess, which is the equal weight array
    x0 = equal_weights, 
    method = 'SLSQP',
    bounds = bounds, 
    constraints = constraints
    )
  # Extract the objective value and append it to the output container
  obj_sd.append(min_result_object['fun'])
# End of for loop
# Convert list to array
obj_sd = np.array(obj_sd)
# Rebind target to a new array object
target = np.linspace(
  start = 0.15, 
  stop = 0.30,
  num = 100
)
```