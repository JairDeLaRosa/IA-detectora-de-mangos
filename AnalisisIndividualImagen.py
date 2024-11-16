import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import seaborn as sns
import pandas as pd
def process_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen al espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de color para mangos en diferentes estados de maduración
    rango_mango = (np.array([0, 0,0]), np.array([255, 255, 255]))  # Ejemplo de rango para mangos 
    
    # Crear máscaras para cada estado de maduración
    mascara_mango = cv2.inRange(hsv_image, rango_mango[0], rango_mango[1])
    
    
    # Aplicar la máscara combinada a la imagen original
    imagen_filtrada = cv2.bitwise_and(image, image, mask=mascara_mango)


    
    # Detectar bordes usando el algoritmo Canny
    bordes = cv2.Canny(imagen_filtrada, 100, 200)
    
    cv2.imshow("Imagen umbralizada", bordes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return bordes

process_image("C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/Ripe/368.jpg")