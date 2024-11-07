import os
import cv2

def eliminar_imagenes_diferente_tamaño(carpeta, tamaño=(224, 224)):
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            img = cv2.imread(os.path.join(carpeta, archivo))
            if img is not None:
                if img.shape[:2] != tamaño:
                    os.remove(os.path.join(carpeta, archivo))
                    print(f"Eliminada: {archivo}")

# Ejemplo de uso
carpeta = "C:/Users/Estudiante/Desktop/train/train/Ripe"
eliminar_imagenes_diferente_tamaño(carpeta)