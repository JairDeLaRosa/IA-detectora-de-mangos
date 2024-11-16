import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar los datos desde el archivo Excel
data = pd.read_excel('caracteristicas_mangos.xlsx')

# Seleccionar las variables independientes (X) y la variable dependiente (y)
X = data[['Suma_Intensidad', 'Promedio_Intensidad', 'Variacion_Intensidad']]
y = data['Etiqueta']

# Convertir las etiquetas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión clasificador
model = DecisionTreeClassifier(random_state=42)

# Optimización de hiperparámetros
param_grid = {
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Hacer predicciones con el conjunto de prueba
y_pred = best_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Mostrar la importancia de las características
feature_importances = best_model.feature_importances_
for name, importance in zip(data.columns[:3], feature_importances):
    print(f'{name}: {importance}')
    
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión en texto
print("Matriz de confusión:")
print(cm)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
disp.ax_.set_title("Matriz de Confusión")
plt.show()    