#
# Abby Donis - 22440
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#---------------------------
# 1. Modelo KNN de SalePrice
#---------------------------
sale = pd.read_csv('train.csv')


X = sale.drop(columns=['SalePrice'])
y = sale['SalePrice']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state=42)

#NA y preprocesor
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  #Mediana
        ('scaler', StandardScaler())
    ]), numerical_columns),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  #moda
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_columns)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsRegressor())
])

pram_tune = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

grid_search = GridSearchCV(pipeline, pram_tune, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
knn_tuned = grid_search.best_estimator_

y_pred = knn_tuned.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE KNN = {round(mae, 3)}")
print(f"RMSE KNN = {round(rmse, 3)}")
print(f"R² KNN = {round(r2, 3)}")