import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('pre-processamento2.csv')
full_dataset = pd.read_csv('house_train.csv')


test = pd.read_csv('pre-processamento_test2.csv')
test_full = pd.read_csv('house_test.csv')

#AJEITANDO DATASET
dataset.columns = range(len(dataset.columns))
dataset = dataset.drop(0, axis=1)
dataset.columns = range(len(dataset.columns))
saleprice = full_dataset['SalePrice']

test.columns = range(len(test.columns))
test = test.drop(0, axis=1)
test.columns = range(len(test.columns))


#importando saleprice
saleprice = pd.Series(np.log(saleprice))



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, saleprice, test_size = 0.2, random_state = 0)


#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators = 100, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))


from sklearn.metrics import r2_score
print('Random Forest r-score', r2_score(y_test, y_pred))


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

#LASSO
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1000))
lasso.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Lasso r-score', r2_score(y_test, y_pred))



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score, cross_val_predict, check_cv
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)
predict = cross_val_predict(estimator = regressor, X = X_train, y = y_train, cv = 5)
print(accuracies)
print(accuracies.mean())


prediction = lasso.predict(test)


print(test_full['Id'])
print(prediction)

prediction2 = pd.DataFrame({'Id':list(test_full['Id']), 'SalePrice':np.exp(prediction)})

print(prediction2)


#prediction2.to_csv('evaluate.csv', index=False)


