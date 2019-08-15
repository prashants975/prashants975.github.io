---
title: "Haberman's Dataset E.D.A"
date: 2019-08-03
tags: [EDA, data science, visualization]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "EDA, Visualization"
mathjax: "true"
---

# Exploratory Data Analysis On Haberman Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Haberman Dataset
Features of the Haberman Datasets are:<br>
**Age** : Age of the Patient<br>
**Op_Year** : Operation Year<br>
**axil_nodes** : Number of axillary lymph nodes. Breast cancer usually spreads to the **axillary lymph nodes** before those at any other location.<br>
**Status** : Survived Or Not. **Class Label**.<br>
    1 For Survived<br>
    2 for Not Survived.<br>


```python
haberman = pd.read_csv('haberman.csv', header = None, names = ['Age','Year', 'axil_nodes', 'Status'] )
haberman.head()
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
      <th>Year</th>
      <th>axil_nodes</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>64</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>62</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>59</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>65</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
haberman.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 306 entries, 0 to 305
    Data columns (total 4 columns):
    Age           306 non-null int64
    Year          306 non-null int64
    axil_nodes    306 non-null int64
    Status        306 non-null int64
    dtypes: int64(4)
    memory usage: 9.6 KB



```python
haberman.describe()
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
      <th>Year</th>
      <th>axil_nodes</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
      <td>306.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.457516</td>
      <td>62.852941</td>
      <td>4.026144</td>
      <td>1.264706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.803452</td>
      <td>3.249405</td>
      <td>7.189654</td>
      <td>0.441899</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>44.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>52.000000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.750000</td>
      <td>65.750000</td>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>69.000000</td>
      <td>52.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#central tendencies when survived
haberman[haberman['Status'] == 1 ].describe()
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
      <th>Year</th>
      <th>axil_nodes</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>225.000000</td>
      <td>225.000000</td>
      <td>225.000000</td>
      <td>225.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52.017778</td>
      <td>62.862222</td>
      <td>2.791111</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.012154</td>
      <td>3.222915</td>
      <td>5.870318</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>43.000000</td>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>52.000000</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>69.000000</td>
      <td>46.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#central tendencies when not survived
haberman[haberman['Status'] == 2 ].describe()
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
      <th>Year</th>
      <th>axil_nodes</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.679012</td>
      <td>62.827160</td>
      <td>7.456790</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.167137</td>
      <td>3.342118</td>
      <td>9.185654</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>34.000000</td>
      <td>58.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>46.000000</td>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>53.000000</td>
      <td>63.000000</td>
      <td>4.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>65.000000</td>
      <td>11.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83.000000</td>
      <td>69.000000</td>
      <td>52.000000</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.median(haberman[haberman['Status'] == 2]['axil_nodes'])
```




    4.0




```python
print('Total Number of patients who survived or not:\n',haberman['Status'].value_counts())
print('Percentage of patients who survived or not:\n',haberman['Status'].value_counts(1))
```

    Total Number of patients who survived or not:
     1    225
    2     81
    Name: Status, dtype: int64
    Percentage of patients who survived or not:
     1    0.735294
    2    0.264706
    Name: Status, dtype: float64



```python
#pearson correlationn
haberman.corr()['Status']
```




    Age           0.067950
    Year         -0.004768
    axil_nodes    0.286768
    Status        1.000000
    Name: Status, dtype: float64



### Basic Inference of the data
The dataset is **little skewed** (not balanced) as the percentage of patient are more for Survived status.<br>
Most of the patients are around age of 52.45(mean) years with max and min as 83 year and 30 years respectively.<br>
Most of the patients are operated in **between 1958 and 1969** with **mean of 1962**.<br>
We can observe that the mean of **number of axil_nodes** for **survived cases is 2.79** and for **not survived cases is 7.45** where as the **median is 0 and 4.00 for survived and not survived** respectively.<br>
Only **axil_nodes have shown some(0.28) correlation** with Status of patients.

## Visualising the Data
### Univariate Analysis
First we will be doing analysis of each of the variables to understand their correlation with our class label(Status).


```python
columns = list(haberman.columns)
features = columns[0:3]
label = columns[-1]
```


```python
haberman_features = haberman.loc[:,features]
haberman_labels = haberman.loc[:,label]
```


```python
plt.figure(1, figsize = (14,5))

for i in range(len(features)):
    plt.subplot(1,3,i+1)
    sns.distplot(haberman[haberman['Status'] == 1][features[i]], label= 'Survived')
    sns.distplot(haberman[haberman['Status'] == 2][features[i]], label= 'Not Survived')
    plt.legend()

plt.subplot(1,3,2)

plt.title('Histogram and density plot of Haberman\'s feature', fontdict = {'fontsize': 20,'fontweight' : 2})
plt.show()



```

    C:\xampp\Py\lib\site-packages\matplotlib\figure.py:98: MatplotlibDeprecationWarning:
    Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      "Adding an axes using the same arguments as a previous axes "


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/haberman/output_14_1.png" alt="">




```python
fig = plt.figure(1, figsize = (15,6))

for i in range(len(features)):
    plt.subplot(1,3,i+1)
    sns.violinplot(data = haberman,x = 'Status', y = features[i])
    plt.title('Voilin Plot for '+ features[i])


```

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/haberman/output_15_0.png" alt="">



We can see that **Age is following a normal distribution**.<br>
**Year is some what following uniform distribution with some spikes** (more operation) in some years.<br>
The **axil nodes feature is very skewed** and its peak is around 0 for **survived patients** and for the **non-survived patients** the density estimate is shigted toward right.<br>
Majority of patient have **less than 5 lymph nodes affected as observed** from mean and violin plot.<br>


```python
plt.figure(1, figsize = (14,8))

for i in range(len(features)):
    plt.subplot(1,3,i+1)
    sns.distplot(haberman[haberman['Status'] == 1][features[i]],hist=False, label= 'Survived PDF')
    sns.distplot(haberman[haberman['Status'] == 1][features[i]],hist=False, label= 'Survived CDF', kde_kws=dict(cumulative=True))
    sns.distplot(haberman[haberman['Status'] == 2][features[i]],hist=False, label= 'Not Survived PDF')
    sns.distplot(haberman[haberman['Status'] == 2][features[i]],hist=False, label= 'Not Survived CDF', kde_kws=dict(cumulative=True))
    plt.legend()

plt.subplot(1,3,2)

plt.title('PDF and CDF plot of Haberman\'s feature', fontdict = {'fontsize': 20,'fontweight' : 2})
plt.show()



```

    C:\xampp\Py\lib\site-packages\matplotlib\figure.py:98: MatplotlibDeprecationWarning:
    Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      "Adding an axes using the same arguments as a previous axes "


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/haberman/output_17_1.png" alt="">



The overlap is huge in pdf and cdf so creating a simple decision boundary with one variable which can classify if a patient will survive or not is not possible with just visualization.


```python
haberman['Status'] = haberman['Status'].map({1:"yes", 2:"no"})
```


```python
haberman.head()
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
      <th>Year</th>
      <th>axil_nodes</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>64</td>
      <td>1</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>62</td>
      <td>3</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>65</td>
      <td>0</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>59</td>
      <td>2</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>65</td>
      <td>4</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>



### Bivariate Analysis


```python
#pairplot for bivariate analysis
sns.pairplot(haberman, hue='Status',height = 3, plot_kws = {'alpha': 0.6})
```




    <seaborn.axisgrid.PairGrid at 0x96bd6720b8>



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/haberman/output_22_1.png" alt="">


Not much can be concluded from the above pair plots as for both the cases **we are finding similar distribution** only in **axil_nodes Vs Age** we can see that **patients with higher age and axil_nodes have less chance for surving as there are not many points that are clustered toward left(more towrd right i.e. high age)**.

## Observations
<ul>
    <li>From the univariate we can say that age and Year don't have much role in whether one will survived or not after being operated.</li>
    <li>Scatter plot of Age and axil_nodes have shown that with <b>higher age and higher number of lymph nodes may have higher chance of not surviving</b>(very small not much correlation). </li>
    <li><b>Higher number of axil_nodes</b> may lead to higher chance of <b>not surving</b>(low correlation).
</ul>
