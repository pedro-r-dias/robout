# content of test_robout.py
import numpy as np
import pandas as pd
import time

df = pd.read_csv(".\\tests\\testSample.csv")

def transformNor(df, normalization=1):
    import robout as rbt
    rs = rbt.robout_scaler(normalization=normalization, ignore=["time"])
    return rs.fit_transform(df).head(20)

def transformStd(df, transform=False):
    import robout as rbt
    rs = rbt.robout_scaler(normalization=0, ignore=["time"])
    fitted = rs.fit_transform(df).head(20)
    if transform:
        return rs.transform(df.head(20))
    else:
        return fitted.head(20)

def invTransformStd(df,dfnorm):
    import robout as rbt
    rs = rbt.robout_scaler(normalization=0, ignore=["time"])
    rs.fit_transform(df)
    return rs.inverse_transform(dfnorm.head(20))

##############################################################################

start_time = time.time()

def test_roboutImport():
    """
    test the robout import
    """
    try:
        import robout as rbt
        assert True
    except:
        assert False

def test_ignoreStr_answer():
    """
    test if string columns or those that are set in the ignore parameter (first 
    two in test sample), are in fact ignored.
    """
    df1 = transformNor(df, normalization=1).iloc[:,:2]
    df2 = df.iloc[:20,:2]
    pd.testing.assert_frame_equal(df1,df2, 
                                  check_dtype=False, check_exact=False)

def test_transformNor_answer():
    """
    test the scaling result when the normalization parameter is 1.
    """
    df1 = transformNor(df, normalization=1).iloc[:,2:]
    df2 = pd.read_csv(".\\tests\\norTestSample.csv").iloc[:,2:]
    pd.testing.assert_frame_equal(np.round((df1-df1.mean())/df1.std(),1), 
                                  np.round((df2-df1.mean())/df2.std(),1), 
                                  check_dtype=False, check_exact=False)
    
def test_transformNor2_answer():
    """
    test the scaling result when the normalization parameter is 2.
    """
    df1 = transformNor(df, normalization=2).iloc[:,2:]
    df2 = pd.read_csv(".\\tests\\nor2TestSample.csv").iloc[:,2:]
    pd.testing.assert_frame_equal(np.round((df1-df1.mean())/df1.std(),1), 
                                  np.round((df2-df1.mean())/df2.std(),1), 
                                  check_dtype=False, check_exact=False)

def test_fitTransformStd_answer():
    """
    test the scaling result when the standardization parameter is true and 
    the fit_transform method output.
    """
    df1 = transformStd(df).iloc[:,2:]
    df2 = pd.read_csv(".\\tests\\stdTestSample.csv").iloc[:,2:]
    pd.testing.assert_frame_equal(np.round((df1-df1.mean())/df1.std(),1), 
                                  np.round((df2-df1.mean())/df2.std(),1), 
                                  check_dtype=False, check_exact=False)

def test_transformStd_answer():
    """
    test the scaling result when the standardization parameter is true.
    """
    df1 = transformStd(df).iloc[:,2:]
    df2 = transformStd(df, transform=True).iloc[:,2:]
    pd.testing.assert_frame_equal(np.round((df1-df1.mean())/df1.std(),1), 
                                  np.round((df2-df1.mean())/df2.std(),1), 
                                  check_dtype=False, check_exact=False)

def test_invTransformStd_answer():
    """
    test the inverse_transform method output.
    """    
    df1 = invTransformStd(df,pd.read_csv(".\\tests\\stdTestSample.csv")).iloc[:,2:]
    df2 = pd.read_csv(".\\tests\\recTestSample.csv").iloc[:,2:]
    pd.testing.assert_frame_equal(np.round((df1-df1.mean())/df1.std(),1), 
                                  np.round((df2-df1.mean())/df2.std(),1), 
                                  check_dtype=False, check_exact=False)

def test_time():
    assert (time.time() - start_time)<30