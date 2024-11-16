import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Cargar los datos desde el archivo CSV
data = pd.read_excel('caracteristicas_mangos.xlsx')

print(data)

# Seleccionar las variables independientes (X) y la variable dependiente (y)
# Supongamos que 'mean_gray_value' es la variable dependiente
X = data[['Suma_Intensidad', 'Promedio_Intensidad', 'Variacion_Intensidad']]
y = data['Etiqueta']

# Convertir las etiquetas a valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión en texto
print("Matriz de confusión:")
print(cm)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
disp.ax_.set_title("Matriz de Confusión")
plt.show()


