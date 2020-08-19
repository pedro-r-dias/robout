
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

table1.table

table2.table

table3.table

table4.table