import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import seaborn as sns
import pandas as pd

# Directorio de las imágenes de mangos
data_path_maduros = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/Ripe"
data_path_podridos = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/Rotten"

# Muestra algunas imágenes de cada clase
# def mostrar_imagenes(clase_path, titulo, n=5):
#     fig, axes = plt.subplots(1, n, figsize=(15, 5))
#     fig.suptitle(titulo, fontsize=16)
#     for i, filename in enumerate(os.listdir(clase_path)[:n]):
#         img_path = os.path.join(clase_path, filename)
#         img = Image.open(img_path)
#         axes[i].imshow(img)
#         axes[i].axis('off')

# # Mostrar 5 imágenes de cada categoría
# mostrar_imagenes(data_path_maduros, 'Mangos Maduros')
# mostrar_imagenes(data_path_podridos, 'Mangos Podridos')
# plt.show()




def process_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen al espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de color para mangos en diferentes estados de maduración
    rango_mango = (np.array([0, 0, 0]), np.array([255, 255, 255]))  # Ejemplo de rango para mangos 
    
    # Crear máscaras para cada estado de maduración
    mascara_mango = cv2.inRange(hsv_image, rango_mango[0], rango_mango[1])
    
    
    # Aplicar la máscara combinada a la imagen original
    imagen_filtrada = cv2.bitwise_and(image, image, mask=mascara_mango)

    
    # Detectar bordes usando el algoritmo Canny
    bordes = cv2.Canny(imagen_filtrada, 100, 200)
    
    # cv2.imshow("Imagen umbralizada", bordes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return bordes
        
    
    
# process_image("C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRipe/65.jpg")
# process_image("C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRotten/20.jpg")

    
    
def extraer_caracteristicas_intensidad_procesada(img_path):
    # Preprocesar la imagen
    img_procesada = process_image(img_path)

    # Extraer características de intensidad
    suma_intensidad = np.sum(img_procesada)/100000
    promedio_intensidad = np.mean(img_procesada)
    variacion_intensidad = np.std(img_procesada)  # Desviación estándar como medida de variación

    return [suma_intensidad, promedio_intensidad, variacion_intensidad]

# Extraer características para cada imagen procesada y crear el DataFrame
caracteristicas = []
etiquetas = []

for filename in os.listdir(data_path_maduros):
    img_path = os.path.join(data_path_maduros, filename)
    caracteristicas.append(extraer_caracteristicas_intensidad_procesada(img_path))
    etiquetas.append("maduro")

for filename in os.listdir(data_path_podridos):
    img_path = os.path.join(data_path_podridos, filename)
    caracteristicas.append(extraer_caracteristicas_intensidad_procesada(img_path))
    etiquetas.append("podrido")

# Crear DataFrame con las características y etiquetas
df = pd.DataFrame(caracteristicas, columns=["Suma_Intensidad", "Promedio_Intensidad", 
                                            "Variacion_Intensidad"])
df["Etiqueta"] = etiquetas


# Mostrar las primeras filas del DataFrame
print(df.head())
def guardar_en_excel(df, nombre_archivo):
    # Guardar el DataFrame en un archivo Excel
    df.to_excel(nombre_archivo, index=False)  # index=False para no guardar el índice como columna
    print(f"El archivo se ha guardado exitosamente en: {nombre_archivo}")

# Llamar a la función para guardar los datos en un archivo Excel
guardar_en_excel(df, 'caracteristicas_mangos.xlsx')



def contar_imagenes(clase_path):
    return len(os.listdir(clase_path))

num_maduros = contar_imagenes(data_path_maduros)
num_podridos = contar_imagenes(data_path_podridos)

# Visualizar distribución
plt.figure(figsize=(6, 4))
plt.bar(['Maduros', 'Podridos'], [num_maduros, num_podridos], color=['green', 'brown'])
plt.title('Distribución de imágenes por clase')
plt.ylabel('Cantidad de imágenes')
plt.show()


# Boxplot para la suma de intensidad
plt.figure(figsize=(7, 4))
sns.boxplot(x='Etiqueta', y='Suma_Intensidad', data=df, palette=['green', 'brown'])
plt.title('Distribución de la Suma de Intensidad por Clase')
plt.show()

# Boxplot para el promedio de intensidad
plt.figure(figsize=(7, 4))
sns.boxplot(x='Etiqueta', y='Promedio_Intensidad', data=df, palette=['green', 'brown'])
plt.title('Distribución del Promedio de Intensidad por Clase')
plt.show()

# Boxplot para la variación de intensidad
plt.figure(figsize=(7, 4))
sns.boxplot(x='Etiqueta', y='Variacion_Intensidad', data=df, palette=['green', 'brown'])
plt.title('Distribución de la Variación de Intensidad por Clase')
plt.show()

# Gráfico de dispersión para Suma de Intensidad vs Variación de Intensidad
plt.figure(figsize=(7, 4))
sns.scatterplot(x='Suma_Intensidad', y='Variacion_Intensidad', hue='Etiqueta', data=df, palette=['green', 'brown'])
plt.title('Suma de Intensidad vs Variación de Intensidad')
plt.show()

# Gráfico de dispersión para Promedio de Intensidad vs Variación de Intensidad
plt.figure(figsize=(7, 4))
sns.scatterplot(x='Promedio_Intensidad', y='Variacion_Intensidad', hue='Etiqueta', data=df, palette=['green', 'brown'])
plt.title('Promedio de Intensidad vs Variación de Intensidad')
plt.show()

# Histograma para la Suma de Intensidad
plt.figure(figsize=(7, 4))
sns.histplot(df, x='Suma_Intensidad', hue='Etiqueta', element='step', palette=['green', 'brown'], kde=True)
plt.title('Histograma de la Suma de Intensidad')
plt.show()

# Histograma para el Promedio de Intensidad
plt.figure(figsize=(7, 4))
sns.histplot(df, x='Promedio_Intensidad', hue='Etiqueta', element='step', palette=['green', 'brown'], kde=True)
plt.title('Histograma del Promedio de Intensidad')
plt.show()

# Histograma para la Variación de Intensidad
plt.figure(figsize=(7, 4))
sns.histplot(df, x='Variacion_Intensidad', hue='Etiqueta', element='step', palette=['green', 'brown'], kde=True)
plt.title('Histograma de la Variación de Intensidad')
plt.show()