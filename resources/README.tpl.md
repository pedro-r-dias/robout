# Robout scaler

> Robust scaling for numeric data with outliers

Welcome! This repository contains the code implementing a scaler preserving the outliers 
found in the unscaled data. It does not discard any outliers, it transforms them to an 
acceptable proximity in relation to the higher density region in the scaled distribution. 
It does that by applying sigmoid transformation after data rescaling using the [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): 
>(x-median)/(percentile(uppq)-percentile(lowq)
 
Thus, between *lowq* and *uppq* parameters, this scaling preserves linearity outsite it 
makes a non-linear transformation pushing the outliers to the linear region. Lastly if 
*standardization* parameter is set to **True**, the data is centered and standard deviation is 
forced to 1.

Follows a small sample (first 5rows x 8cols) from the [testSample.csv](./tests/testSample.csv) file and a 
violin plot of the first 8 variables before scaling.

table1.table

![Violin plots of scaled test data](./resources/fig1.png)


After the robout scaling (without standardization) the same sample and the violin plots look as follows:

table2.table

![Violin plots of scaled test data](./resources/fig2.png)

After applying standardization, this is the result:

table3.table

![Violin plots of scaled and standardized test data](./resources/fig3.png)


## Installation

```sh
pip install -i https://test.pypi.org/simple/ robout-pedro-r-dias
```

## Usage example

Follows a data scaling example using *fit_transform* (robout also includes the *transform* method). 

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import robout as rbt

df = pd.read_csv(".\\tests\\testSample.csv")

rs = rbt.robout_scaler(standardization=True, ignore=["time"])
scaled = rs.fit_transform(df)

f, ax = plt.subplots(figsize=(20, 5))
sns.violinplot(data=scaled.iloc[:,2:10])
```

![Violin plots of scaled and standardized test data](./resources/fig3.png)

To revert the scaling use the inverse_transform method as follows:

```python
unscaled = rs.inverse_transform(scaled)
unscaled.iloc[:5,:10]
```

table4.table


## Release History

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
