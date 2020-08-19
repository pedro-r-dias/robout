class robout_scaler:
    """
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
    """
                           
    def __init__(self, uppq=0.9, lowq=0.1, standardization=True, ignore=[]):
        """
        Parameters
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
        """
        self.uppq=uppq
        self.lowq=lowq
        self.standardization=standardization
        self.ignore=ignore
        
    def fit_transform(self, df):
        """
        This transformation preserves outliers found in the unscaled data 
        as outliers also in the scaled data, but transforms them to an 
        acceptable proximity in relation to the higher density region 
        in the scaled distribution. It does that by applying sigmoid 
        transformation after data rescaling using the RobustScaler:  
               (x-median)/(percentile(uppq)-percentile(lowq).
        Thus, between lowq and uppq, this scaling preserves linearity.
        Lastly if standardization==True, the data is centered and standard 
        deviation is set to 1.
    
        It returns the scaled input dataframe and functions to scale and 
        unscale dataframes transforming the data to its original units.
        When unscaling, inf and -inf values are transformed back to, 
        respectively, the median of those greater than the uppq percentile 
        and the median of those lower than the lowq percentile.
        
        Executing the fit_transform also generates the logic of the transform 
        and inverse_transform methods

        Parameters
        ----------
        df : pandas dataframe or numpy 2d array.
             The unscaled, original input data. It can include string type
             columns or other columns to be excluded from scaling (using ignore 
             parameter). All other columns will be transformed according to the 
             parameterization.

        Returns
        ------
        scaled pandas dataFrame. 
        
        """
        import pandas as pd
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # ensure input df is a pandas dataframe or numpy ndarray
        returnnp = True
        if type(df) is np.ndarray:
            try:
                df = pd.DataFrame(df)
            except:
                raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
        elif type(df) is pd.DataFrame:
            returnnp = False
        else:
            raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
        
        # store descriptive stats about each column
        med = df.median()
        upp = df.quantile(self.uppq)
        low = df.quantile(self.lowq)
        pif = df.apply(lambda v: v[v>upp[v.name]].median() if v.name in upp.index else np.nan)
        nif = df.apply(lambda v: v[v<low[v.name]].median() if v.name in low.index else np.nan)
        stg = df.apply(lambda v: v.apply(type).eq(str).any() or v.name in self.ignore)

        # Returns for each column name a lambda function converting type to 
        # int if all unscaled values are int and replace inf and -inf values 
        # by the, respectively, 99.9% percentile and 0.1% percentile.
        def intType(n):
            if stg[n]:
                # ignoring strings
                return lambda v: v 
            elif np.array_equal(df[n], df[n].astype(int)):
                return lambda v: \
                       v.replace(np.inf, pif[n]).replace(-np.inf, nif[n])\
                        .astype(int)
            else:
                return lambda v: \
                       v.replace(np.inf, pif[n]).replace(-np.inf, nif[n])\
                        .astype(np.float)

        # A dictionary (column names as keys) with the lambda function 
        # returned by the intType function.    
        convInt = pd.Series(df.columns, index=df.columns).apply(intType)

        # apply normalization to df
        dfn = df.apply(lambda v: v if stg[v.name] else \
                       1/(1+np.exp(-(v.astype(np.float)-med[v.name])/(upp[v.name]-low[v.name]))), 
                       axis=0)

        # Get the descriptive stats of the so far normalized columns 
        # needed for the normalization step that makes mean=0 and std=1.
        if self.standardization:
            mea = dfn.mean()
            std = dfn.std()
            # apply standardization to df
            dfn = dfn.apply(lambda v: v if stg[v.name] else \
                            (v-mea[v.name])/std[v.name], 
                            axis=0)
        else:
            mea = pd.Series(0, index=df.columns)
            std = pd.Series(1, index=df.columns)

        # Generating the lambda functions (one for each column) used 
        # by the generic function that will scale any dataframes.
        def scalerGen(n):
            if stg[n]:
                return lambda v: v
            else:
                return lambda v: \
                       (1/(1+np.exp(-(v.astype(np.float)-med[n])/(upp[n]-low[n])))-mea[n])/std[n]

        # Generating the lambda functions (one for each column) used 
        # by the generic function that will unscale any scaled dataframes. 
        def unscalerGen(n):
            if stg[n]:
                return lambda v: v
            else:
                return lambda v: \
                       convInt[n](med[n]-(upp[n]-low[n])*np.log(1/(v*std[n]+mea[n])-1))

        # Crete series storing the lambda functions dedicated to each column
        scalein = pd.Series(df.columns, index=df.columns).apply(scalerGen)
        scaleout = pd.Series(df.columns, index=df.columns).apply(unscalerGen)

        # Create the functions that will be returned by the scaler as methods to 
        # transform (scaling/unscaling) any data sharing columns with df.
        def scale(data):
            """
            Transformation scaling the data according to the parameterization of
            the robout_scaler instance.
            
            Parameters
            ----------
            data : pandas dataframe or numpy 2d array.
                 The unscaled, original input data. It can include string type
                 columns or other columns to be excluded from scaling (using ignore 
                 parameter). All other columns will be transformed according to the 
                 parameterization.

            Returns
            ------
            scaled pandas dataFrame. 

            """
            # ensure input df is a pandas dataframe or numpy ndarray
            returnnp = True
            if type(data) is np.ndarray:
                try:
                    data = pd.DataFrame(data)
                except:
                    raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
            elif type(data) is pd.DataFrame:
                returnnp = False
            else:
                raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
            data = data.apply(lambda v: scalein[v.name](v))
            if returnnp:
                return data.values
            else:
                return data  
        self.transform = scale
        
        def unscale(data):
            """
            Inverse transformation to go back to the original units.
            When unscaling, inf and -inf values are transformed back to, 
            respectively, the median of those greater than the uppq percentile 
            and the median of those lower than the lowq percentile.
        
            Parameters
            ----------
            data : pandas dataframe or numpy 2d array.
                 Scaled data. It can include string type columns or other columns 
                 to be excluded from scaling (using ignore parameter). All other 
                 columns will be transformed according to the parameterization.

            Returns
            ------
            pandas dataFrame having the columns transformed back to the original units. 

            """
            # ensure input df is a pandas dataframe or numpy ndarray
            returnnp = True
            if type(data) is np.ndarray:
                try:
                    data = pd.DataFrame(data)
                except:
                    raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
            elif type(data) is pd.DataFrame:
                returnnp = False
            else:
                raise ValueError("Input must be a 2d numpy array or a pandas DataFrame")
            data = data.apply(lambda v: scaleout[v.name](v))
            if returnnp:
                return data.values
            else:
                return data
        self.inverse_transform = unscale

        if returnnp:
            return dfn.values
        else:
            return dfn