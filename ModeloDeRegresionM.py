import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Cargar los datos
data = pd.read_excel('caracteristicas_mangos.xlsx')

# Seleccionar las variables independientes (X) y la variable dependiente (y)
X = data[['Suma_Intensidad', 'Promedio_Intensidad', 'Variacion_Intensidad']]
y = data['Etiqueta']

# Convertir las etiquetas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Escalar las variables predictoras
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Distribución de clases después del balanceo:")
print(pd.Series(y_train_balanced).value_counts())

# # Crear un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_balanced, y_train_balanced)
# Realizar predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calcular métricas
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("MSE (entrenamiento):", mse_train)
print("MSE (prueba):", mse_test)
print("R² (entrenamiento):", r2_train)
print("R² (prueba):", r2_test)


# # Crear un pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('regression', Ridge())
# ])

# # Configurar GridSearch
# param_grid = {
#     'regression__alpha': [0.01, 0.1, 1, 10]  # Parámetro de regularización
# }
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train, y_train)

# # Resultados
# print("Mejor R²:", grid_search.best_score_)
# print("Mejor configuración:", grid_search.best_params_)


# Optimización con validación cruzada
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# Predicción en el conjunto de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión en texto
print("Matriz de confusión:")
print(cm)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
disp.ax_.set_title("Matriz de Confusión")
plt.show()