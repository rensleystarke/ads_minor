
# 9 Evaluate

So now we have prepared our data for our final analysis. Naturally, we are curious to see how much it matters if we properly explore and prepare our data. In this small example, we will compare the accuracy of a simple linear regression algorithm that was learned on the original data and one that is learned on the transformed data.


```python
import pandas as pd
import numpy as np
import warnings
import statsmodels.formula.api as sm
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('train.csv') # use training set to learn a prediction model
df_test = pd.read_csv('test.csv') # use a separate test set to evaluate the accuracy of your model
```


```python
#applying log transformation
df['SalePriceL'] = np.log(df.SalePrice)
df['GrLivAreaL'] = np.log(df.GrLivArea)
df_test['GrLivAreaL'] = np.log(df_test.GrLivArea)
```

Fit a linear line to the data, to optimally predict SalePrice given a value for GrLivArea.


```python
model = sm.ols(formula="SalePrice ~ GrLivArea", data=df).fit()
```

Use the learned model (line) to predict salesprices for the cases in the test set.


```python
df_test['SalePrice'] = model.predict(df_test)
```


```python
testraw = df_test[['Id', 'SalePrice']]
```

The test set does not contain the actual sales prices, so we have to submit the file to Kaggle to get an evaluation metric. Write Id and SalePrice to a CSV file, that can be submitted on Kaggle.


```python
df_test[['Id', 'SalePrice']].to_csv('testraw', index=False)
```

The score on Kaggle reads: 0.28918

This score is based on the root mean squared error (the most common evaluation metric used for linear regression). Lower is better.

Do the same for the transformed data.


```python
modell = sm.ols(formula="SalePriceL ~ GrLivAreaL", data=df).fit()
```


```python
df_test['SalePrice'] = modell.predict(df_test)
```

Since we tranformed price to log, the model returns log(expected Price), so we have to invert the function to get the actual price. The inverse of a `log` function is an `exponential` function.


```python
df_test.SalePrice = np.exp(df_test.SalePrice)
```


```python
df_test[['Id', 'SalePrice']].to_csv('testtransformed', index=False)
```

The score on Kaggle reads 0.28796


```python

```
