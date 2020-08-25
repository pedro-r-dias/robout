# Robout scaler

> Robust scaling for numeric data with outliers

Welcome! This repository contains the code implementing a scaler preserving outliers 
found in unscaled data. It does not discard outliers, it transforms them to a 
controllable proximity in relation to the higher density region of the scaled distribution. 
It does that by applying sigmoid transformation after an initial data scaling using the 
Robust Scaler as implemented in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): 
*(x-median)/(percentile(uppq)-percentile(lowq)*.
 
Thus, between *lowq* and *uppq* parameters, this scaling preserves linearity, whereas outside, 
it makes a non-linear transformation pushing the outliers to the linear region. 

Lastly, if *normalization* parameter is set to **False** or **0**, the data is centered and 
standard deviation is forced to 1 (the [standard scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 
procedure). If normalization is set to **True** or **1**, the data is normalized to be in
the interval between 0 and 1, else, if it is set to 2, the data is normalized to be in the 
interval between 0 and 2. 

Follows a small sample (first 5rows x 8cols) from the [testSample.csv](./tests/testSample.csv) file and a 
violin plot of the first 8 variables before scaling.

table1.table

![Violin plots of scaled test data](./resources/fig1.png)


After the Robout scaling (without standardization) the same sample and the violin plots look as follows:

table2.table

![Violin plots of scaled test data](./resources/fig2.png)


After applying standardization, this is the result:

table4.table

![Violin plots of scaled and standardized test data](./resources/fig4.png)


## Installation

The installation of Robout scaler requires [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) and [Numpy](https://numpy.org/install/).

The latest release of Robout scaler can be installed from [PyPI](https://pypi.org/project/robout/0.0.1/) using ``pip``:

```sh
    pip install robout
```

Or via conda:

```sh
    conda install -c pedro-r-dias robout
```


## Usage example

Follows a data scaling example using *fit_transform* (Robout scaler also includes the *transform* method). 

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import robout as rbt

df = pd.read_csv(".\\tests\\testSample.csv")

rs = rbt.robout_scaler(normalization=0, ignore=["time"])
scaled = rs.fit_transform(df)

f, ax = plt.subplots(figsize=(20, 5))
sns.violinplot(data=scaled.iloc[:,2:10])
```

![Violin plots of scaled and standardized test data](./resources/fig4.png)

To revert the scaling use the inverse_transform method as follows:

```python
unscaled = rs.inverse_transform(scaled)
unscaled.iloc[:5,:10]
```

table5.table

## Comparison with other scaling methods

### Robout scaler
![Robout scaler](./resources/fig5.png)

### Standard scaler
![Standard scaler](./resources/fig6.png)

### Robust Scaler
![Robust Scaler](./resources/fig7.png)

### MinMax Scaler
![MinMax Scaler](./resources/fig8.png)

### Fibonacci scaler
![Fibonacci scaler](./resources/fig9.png)

## Release History

* 0.1.1
    * Documentation updates
* 0.1.0
    * Ready
* 0.0.1
    * Work in progress

## Meta

Pedro Dias – pedroruivodias@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.


## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
