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
# Modelo KNN de SalePrice
#---------------------------
sale = pd.read_csv('train.csv')

X = sale.drop(columns=['SalePrice'])
y = sale['SalePrice']

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

# Preprocesamiento para regresión
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Mediana
        ('scaler', StandardScaler())
    ]), numerical_columns),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Moda
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

grid_search = GridSearchCV(pipeline, pram_tune, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
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
# Modelo de Clasificación
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

# Preprocesamiento para clasificación
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

# Grid de hiperparámetros para clasificación
param_grid_cls = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

grid_search_cls = GridSearchCV(pipeline_cls, param_grid_cls, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_cls.fit(X_train_cls, y_train_cls_enc)
knn_tuned_cls = grid_search_cls.best_estimator_

# Predicción en el conjunto de prueba
y_pred_cls_enc = knn_tuned_cls.predict(X_test_cls)

# Evaluación de clasificación
accuracy = accuracy_score(y_test_cls_enc, y_pred_cls_enc)
print(f"Accuracy KNN Clasificación = {round(accuracy, 3)}")
print("\nReporte de Clasificación:")
print(classification_report(y_test_cls_enc, y_pred_cls_enc, target_names=le.classes_))

# ---------------------------
# Inciso 8
# ---------------------------
from sklearn.model_selection import cross_val_score

# Validación Cruzada para el modelo de regresión (mejor modelo tunado)
cv_scores_reg = cross_val_score(knn_tuned, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_reg = np.mean(np.sqrt(-cv_scores_reg))
print("\nValidación Cruzada - Regresión")
print("CV RMSE:", round(cv_rmse_reg, 3))

# Validación Cruzada para el modelo de clasificación (mejor modelo tunado)
cv_scores_cls = cross_val_score(knn_tuned_cls, X_train_cls, y_train_cls_enc, cv=5, scoring='accuracy')
print("\nValidación Cruzada - Clasificación")
print("CV Accuracy:", round(np.mean(cv_scores_cls), 3))

# ---------------------------
# Inciso 9
# ---------------------------
# Para regresión: Modelo Base sin tuning
base_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('knn', KNeighborsRegressor())
])
base_pipeline.fit(X_train, y_train)
y_pred_base = base_pipeline.predict(X_test)
mae_base = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)
print("\nModelo Base (Regresión) - Sin tuning")
print(f"MAE Base = {round(mae_base, 3)}")
print(f"RMSE Base = {round(rmse_base, 3)}")
print(f"R² Base = {round(r2_base, 3)}")
# Nota: Los hiperparámetros tunables en KNN son: n_neighbors, weights y p.

# Para clasificación: Modelo Base sin tuning
base_pipeline_cls = Pipeline([
    ('preprocessing', preprocessor_cls),
    ('knn', KNeighborsClassifier())
])
base_pipeline_cls.fit(X_train_cls, y_train_cls_enc)
y_pred_base_cls = base_pipeline_cls.predict(X_test_cls)
accuracy_base = accuracy_score(y_test_cls_enc, y_pred_base_cls)
print("\nModelo Base (Clasificación) - Sin tuning")
print(f"Accuracy Base = {round(accuracy_base, 3)}")

# ---------------------------
# Inciso 10
# ---------------------------
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Para Naive Bayes, convertimos la salida de OneHotEncoder a denso.
from sklearn.preprocessing import FunctionTransformer
to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

# Medición de tiempos para el modelo KNN (clasificación tunado)
start = time.time()
knn_tuned_cls.fit(X_train_cls, y_train_cls_enc)
knn_train_time = time.time() - start
start = time.time()
_ = knn_tuned_cls.predict(X_test_cls)
knn_pred_time = time.time() - start

# Árbol de Decisión
pipeline_dt = Pipeline([
    ('preprocessing', preprocessor_cls),
    ('dt', DecisionTreeClassifier(random_state=42))
])
start = time.time()
pipeline_dt.fit(X_train_cls, y_train_cls_enc)
dt_train_time = time.time() - start
start = time.time()
y_pred_dt = pipeline_dt.predict(X_test_cls)
dt_pred_time = time.time() - start
dt_accuracy = accuracy_score(y_test_cls_enc, y_pred_dt)

# Random Forest
pipeline_rf = Pipeline([
    ('preprocessing', preprocessor_cls),
    ('rf', RandomForestClassifier(random_state=42))
])
start = time.time()
pipeline_rf.fit(X_train_cls, y_train_cls_enc)
rf_train_time = time.time() - start
start = time.time()
y_pred_rf = pipeline_rf.predict(X_test_cls)
rf_pred_time = time.time() - start
rf_accuracy = accuracy_score(y_test_cls_enc, y_pred_rf)

# Naive Bayes
pipeline_nb = Pipeline([
    ('preprocessing', preprocessor_cls),
    ('to_dense', to_dense),
    ('nb', GaussianNB())
])
start = time.time()
pipeline_nb.fit(X_train_cls, y_train_cls_enc)
nb_train_time = time.time() - start
start = time.time()
y_pred_nb = pipeline_nb.predict(X_test_cls)
nb_pred_time = time.time() - start
nb_accuracy = accuracy_score(y_test_cls_enc, y_pred_nb)

print("\nComparación de modelos de clasificación:")
print(f"KNN Tuned: Accuracy = {round(accuracy,3)}, Tiempo de entrenamiento = {knn_train_time:.3f}s, Tiempo de predicción = {knn_pred_time:.3f}s")
print(f"Árbol de Decisión: Accuracy = {round(dt_accuracy,3)}, Tiempo de entrenamiento = {dt_train_time:.3f}s, Tiempo de predicción = {dt_pred_time:.3f}s")
print(f"Random Forest: Accuracy = {round(rf_accuracy,3)}, Tiempo de entrenamiento = {rf_train_time:.3f}s, Tiempo de predicción = {rf_pred_time:.3f}s")
print(f"Naive Bayes: Accuracy = {round(nb_accuracy,3)}, Tiempo de entrenamiento = {nb_train_time:.3f}s, Tiempo de predicción = {nb_pred_time:.3f}s")
# Para ser más preciso se realizó nuevamente los otros métodos en este mismo archivo y con el mismo database
# Sí, fue dolor. Más o menos
