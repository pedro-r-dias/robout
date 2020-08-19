
This scaler preserves outliers found in the unscaled data as 
outliers also in the scaled data, but transforms them to an 
acceptable proximity in relation to the higher density region 
in the scaled distribution. It does that by applying sigmoid 
transformation after data rescaling using the RobustScaler: 
       (x-median)/(percentile(uppq)-percentile(lowq).
Thus, between lowq and uppq, this scaling preserves linearity.
Lastly if standardization==True, the data is centered and standard 
deviation is set to 1.

Attributes
----------
uppq : float
    a value between 0 and 1 defining the upper quantile expected 
    to include outliers.

lowq : float
    a value between 0 and 1 defining the lower quantile expected 
    to include outliers.

standardization : boolean
    a boolean flag indicating if standardization, meaning the
    forcing of the mean to 0 and the standard deviation to 1, should 
    be undertaken.

ignore : list
    list of column names from the input dataframe that shall not be 
    scaled due to whatever reason.

Methods
-------
fit_transform(df)
    It returns the scaled input dataframe and functions to scale and 
    unscale dataframes transforming the data to its original units.
    When unscaling, inf and -inf values are transformed back to, 
    respectively, the median of those greater than the uppq percentile 
    and the median of those lower than the lowq percentile.

    Executing the fit_transform also generates the logic of the transform 
    and inverse_transform methods

transform(data)
    Transformation scaling the data according to the parameterization of
    the robout_scaler instance.

inverse_transform(data)
    Inverse transformation to go back to the original units.
    When unscaling, inf and -inf values are transformed back to, 
    respectively, the median of those greater than the uppq percentile 
    and the median of those lower than the lowq percentile.

Table: sample from table1.csv file. \label{table:table1}

| id                    |       time |    AA |   AB |   AC |    AD |       AE |   AF |
|:----------------------|-----------:|------:|-----:|-----:|------:|---------:|-----:|
| x4425323655333165260  | 1515049200 |  2200 | -118 | -119 | 0.917 | 0.006693 |  845 |
| x10230558070004111555 | 1515060000 |  3197 | -118 | -118 | 0.938 | 0.026903 | 1352 |
| x18350715752638066598 | 1515006000 |    75 | -108 |  -99 | 0.427 | 0.033111 |  149 |
| x10230558070004111555 | 1515150000 |  2967 | -118 | -119 | 0.944 | 0.021121 | 1420 |
| x16587885833987648653 | 1515186000 | 10395 | -119 | -120 | 0.944 | 0.069466 | 1943 |


Table: sample from table2.csv file. \label{table:table2}

| id                    |       time |    AA |    AB |    AC |    AD |    AE |    AF |
|:----------------------|-----------:|------:|------:|------:|------:|------:|------:|
| x4425323655333165260  | 1515049200 | 0.447 | 0.531 | 0.526 | 0.492 | 0.427 | 0.421 |
| x10230558070004111555 | 1515060000 | 0.478 | 0.542 | 0.555 | 0.529 | 0.474 | 0.495 |
| x18350715752638066598 | 1515006000 | 0.383 | 0.964 | 0.998 | 0.032 | 0.489 | 0.325 |
| x10230558070004111555 | 1515150000 | 0.471 | 0.529 | 0.529 | 0.539 | 0.461 | 0.505 |
| x16587885833987648653 | 1515186000 | 0.694 | 0.447 | 0.443 | 0.538 | 0.574 | 0.581 |


Table: sample from table3.csv file. \label{table:table3}

| id                    |       time |     AA |     AB |     AC |     AD |     AE |     AF |
|:----------------------|-----------:|-------:|-------:|-------:|-------:|-------:|-------:|
| x4425323655333165260  | 1515049200 | -0.714 | -0.043 | -0.094 |  0.21  | -0.919 | -0.797 |
| x10230558070004111555 | 1515060000 | -0.398 |  0.037 |  0.107 |  0.5   | -0.496 | -0.117 |
| x18350715752638066598 | 1515006000 | -1.37  |  3.134 |  3.086 | -3.428 | -0.364 | -1.675 |
| x10230558070004111555 | 1515150000 | -0.471 | -0.054 | -0.071 |  0.585 | -0.618 | -0.026 |
| x16587885833987648653 | 1515186000 |  1.786 | -0.659 | -0.651 |  0.574 |  0.401 |  0.675 |


Table: sample from table4.csv file. \label{table:table4}

| id                    |       time |    AA |   AB |   AC |    AD |       AE |   AF |
|:----------------------|-----------:|------:|-----:|-----:|------:|---------:|-----:|
| x4425323655333165260  | 1515049200 |  2200 | -118 | -119 | 0.917 | 0.006693 |  845 |
| x10230558070004111555 | 1515060000 |  3197 | -118 | -118 | 0.938 | 0.026903 | 1352 |
| x18350715752638066598 | 1515006000 |    75 | -108 |  -99 | 0.427 | 0.033111 |  149 |
| x10230558070004111555 | 1515150000 |  2967 | -118 | -119 | 0.944 | 0.021121 | 1420 |
| x16587885833987648653 | 1515186000 | 10395 | -119 | -120 | 0.944 | 0.069466 | 1943 |
