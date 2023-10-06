import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class RegressionModel():
    
    def __init__(self, test_size = 0.25) -> None:
        self.model = RandomForestRegressor() # could be changed if pleased (SGDR Regressor)
        self.test_size = test_size
        self.hyperparameters = None
        pass

    def train_test_split(self, X,y, test_size):

        ''' Implementing train-test split logic '''
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= 0)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

        
        return train_set, test_set
    
    def preprocess(self, X):

        '''Scaling data'''
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X
    
    def fit(self, train_set): 

        ''' Performs training logic for any model. First pre-processing, 
        and then training. '''
        
        X_train, y_train = train_set
        X_train = self.preprocess(X_train)     
        self.model.fit(X_train, y_train)

        return self.model
    
    def evaluate(self, test_set):

        '''Performs prediction of the model and stores metrics'''
        
        X_test, y_test = test_set
        X_test = self.preprocess(X_test)
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_pred= y_pred, y_true= y_test)
        r2 = r2_score(y_pred= y_pred, y_true= y_test)

        return mae, r2
    


