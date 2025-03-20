# Wilson Calderón - 22018
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

# ---------------------------
# 4. Modelo de Clasificación
# ---------------------------

# Creamos la variable categórica 'PriceCat' a partir de 'SalePrice'
sale['PriceCat'] = pd.qcut(sale['SalePrice'], q=3, labels=['barata', 'media', 'cara'])

# Definimos las variables predictoras y la variable respuesta para clasificación.
X_class = sale.drop(columns=['SalePrice', 'PriceCat'])
y_class = sale['PriceCat']

# Dividimos los datos en entrenamiento y prueba (70-30)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# Codificamos las etiquetas de clase a valores numéricos
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_cls_enc = le.fit_transform(y_train_cls)
y_test_cls_enc = le.transform(y_test_cls)

# Preprocesamiento: obtenemos columnas numéricas y categóricas
numerical_columns_cls = X_class.select_dtypes(include=['number']).columns
categorical_columns_cls = X_class.select_dtypes(include=['object']).columns

preprocessor_cls = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_columns_cls),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_columns_cls)
])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Pipeline para clasificación
pipeline_cls = Pipeline([
    ('preprocessing', preprocessor_cls),
    ('knn', KNeighborsClassifier())
])

# Grid de hiperparámetros
param_grid_cls = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

# Ajuste con GridSearchCV usando scoring 'accuracy'
grid_search_cls = GridSearchCV(pipeline_cls, param_grid_cls, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_cls.fit(X_train_cls, y_train_cls_enc)
knn_tuned_cls = grid_search_cls.best_estimator_

# Predicción en el conjunto de prueba
y_pred_cls_enc = knn_tuned_cls.predict(X_test_cls)

# Evaluación
accuracy = accuracy_score(y_test_cls_enc, y_pred_cls_enc)
print(f"Accuracy KNN Clasificación = {round(accuracy, 3)}")
print("\nReporte de Clasificación:")
print(classification_report(y_test_cls_enc, y_pred_cls_enc, target_names=le.classes_))
