import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# Crear un modelo de regresión logística
model = LogisticRegression()

# Optimización con validación cruzada
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

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