
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def LRSelect(X_train, y_train, X_train_drop_list):
    # drop variables of training dataset
    X_train_drop = X_train.drop(X_train_drop_list,axis=1)
    # add constant to the model
    X_train_drop = sm.add_constant(X_train_drop)
    # create linear regression model
    model = sm.OLS(y_train, X_train_drop)
    res = model.fit()

    # For each X, calculate VIF and save in dataframe
    vif = pd.DataFrame().round(1)
    vif["VIF Factor"] = [variance_inflation_factor(X_train_drop.values, i) for i in range(X_train_drop.shape[1])]
    vif["features"] = X_train_drop.columns
    
    # yield the results
    yield res.summary()
    yield vif.round(1)
