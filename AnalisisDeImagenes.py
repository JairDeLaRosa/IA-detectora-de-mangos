import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import statistics
import pandas as pd
import os
from PIL import Image

from skimage.data import page
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def convert_toGrayImage(filename):
    img = ski.io.imread(filename)
    img = ski.util.img_as_ubyte(img)
    
    # Verificar si la imagen tiene 3 canales (RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        gray_image = rgb2gray(img)
    else:
        # Si la imagen ya está en escala de grises, usarla tal cual
        gray_image = img
    
    return gray_image, img

def get_binaryMango(gray_image, sigma=1.0, connectivity = 2):
    blurred_image = ski.filters.gaussian(gray_image, sigma=sigma)
    bin_image = blurred_image > threshold_otsu(blurred_image)
    
    filled_img = binary_fill_holes(bin_image)
    labeled_image, count = ski.measure.label(filled_img, connectivity=connectivity, return_num=True)

    object_features = ski.measure.regionprops(labeled_image)
    object_areas = [objf["area"] for objf in object_features]
    max_size = max(object_areas)

    mango_mask = remove_small_objects(labeled_image, min_size=max_size-1)
    #plt.imshow(mango_mask)
    #plt.show() 

    mango_mask = mango_mask < 1
    return mango_mask

def std_image(gray_img, mango_mask):
    mango_mask = ski.util.invert(mango_mask)
    idx_list = np.where(mango_mask == 1)
    mangoPixel_values = gray_img[idx_list]
    
    # Convertir los valores de píxeles a float para evitar desbordamiento
    mangoPixel_values = mangoPixel_values.astype(np.float64)
    
    stdev = statistics.stdev(mangoPixel_values)
    return stdev

def extract_properties(mango_img, gray_img, mango_mask):

    labeled_mango, count = ski.measure.label(mango_mask, connectivity=2, return_num=True)
    object_features = ski.measure.regionprops(labeled_mango)

    var_color_props = ['mean_intensity']
    color_props = ski.measure.regionprops_table(labeled_mango, mango_img, properties=var_color_props)

    var_gris_props = ['area', 'max_intensity', 'min_intensity', 'mean_intensity']
    gris_props = ski.measure.regionprops_table(labeled_mango, gray_img, properties=var_gris_props)
    
    grayImg_stdev = std_image(gray_img, mango_mask)
    df_image = pd.DataFrame(gris_props)
    
    df_image.rename(columns={'max_intensity': 'max_gray_value', 'min_intensity': 'min_gray_value', 'mean_intensity': 'mean_gray_value'}, inplace=True)
    
    # Verificar si las claves existen en color_props antes de asignarlas
    if 'mean_intensity-0' in color_props:
        df_image['mean_value_R'] = color_props['mean_intensity-0']
    else:
        df_image['mean_value_R'] = np.nan
    
    if 'mean_intensity-1' in color_props:
        df_image['mean_value_G'] = color_props['mean_intensity-1']
    else:
        df_image['mean_value_G'] = np.nan
    
    if 'mean_intensity-2' in color_props:
        df_image['mean_value_B'] = color_props['mean_intensity-2']
    else:
        df_image['mean_value_B'] = np.nan
    
    df_image['std'] = grayImg_stdev 
    
    return df_image

def intensity_table(grayValues):
    table = {}
    for ival in range(256):
        table['val_'+ str(ival)] = [np.count_nonzero(grayValues == ival)]
       
    df_table = pd.DataFrame.from_dict(table, orient='columns')
    return df_table    

def hist2features(grayImg, mangoMask):
    #color = ski.util.img_as_ubyte(colorImg)
    gray = ski.util.img_as_ubyte(grayImg)
    mango_mask = ski.util.invert(mangoMask)

    masked_gray = gray * mango_mask
    plt.imshow(masked_gray, cmap='gray')
    plt.show()
    df_intensities = intensity_table(masked_gray)
    print(df_intensities)
    return df_intensities

folderRipe = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRipe"
folderRotten = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRotten"

def saveData(folderRipe, folderRotten):
    all_results_df = pd.DataFrame()
    
def analizerImages(folder_path):
    all_results_df = pd.DataFrame()
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = folder_path + "/" + filename
            grayImg, img = convert_toGrayImage(file_path)
            mango_mask = get_binaryMango(grayImg)
            resultados = extract_properties(img, grayImg, mango_mask)
            
            all_results_df = pd.concat([all_results_df, resultados], ignore_index=True)
    
    # Guardar los resultados en un archivo CSV
    all_results_df.to_csv('results.csv', index=False)

    # Graficar los resultados en ventanas separadas
    
    # Gráficos de barras
    plt.figure()
    all_results_df['max_gray_value'].plot(kind='hist')
    plt.title('Max Gray Value (Bar Plot)')
    plt.show()

    plt.figure()
    all_results_df['min_gray_value'].plot(kind='hist')
    plt.title('Min Gray Value (Bar Plot)')
    plt.show()

    plt.figure()
    all_results_df['mean_gray_value'].plot(kind='hist')
    plt.title('Mean Gray Value (Bar Plot)')
    plt.show()

    plt.figure()
    all_results_df['std'].plot(kind='hist')
    plt.title('Standard Deviation (Bar Plot)')
    plt.show()
    
    plt.figure()
    all_results_df['area'].plot(kind='line')
    plt.title('Area (Bar Plot)')
    plt.show()
    # Gráficos de caja
    plt.figure()
    all_results_df.boxplot(column=['max_gray_value'])
    plt.title('Max Gray Value (Box Plot)')
    plt.show()

    plt.figure()
    all_results_df.boxplot(column=['min_gray_value'])
    plt.title('Min Gray Value (Box Plot)')
    plt.show()

    plt.figure()
    all_results_df.boxplot(column=['mean_gray_value'])
    plt.title('Mean Gray Value (Box Plot)')
    plt.show()

    plt.figure()
    all_results_df.boxplot(column=['std'])
    plt.title('Standard Deviation (Box Plot)')
    plt.show()
    
    plt.figure()
    all_results_df.boxplot(column=['area'])
    plt.title('Area (Box Plot)')
    plt.show()

    # Gráficos de dispersión
    plt.figure()
    plt.scatter(all_results_df.index, all_results_df['max_gray_value'], label='Max Gray Value')
    plt.scatter(all_results_df.index, all_results_df['min_gray_value'], label='Min Gray Value')
    plt.scatter(all_results_df.index, all_results_df['mean_gray_value'], label='Mean Gray Value')
    plt.scatter(all_results_df.index, all_results_df['std'], label='Standard Deviation')
    plt.legend()
    plt.title('Scatter Plot of Values')
    plt.show()

# analizerImages(folderRipe)            
# analizerImages(folderRotten)

# Cargar los datos desde el archivo CSV
data = pd.read_csv('results.csv')

print(data)

# Seleccionar las variables independientes (X) y la variable dependiente (y)
# Supongamos que 'mean_gray_value' es la variable dependiente
# X = data[['max_gray_value', 'min_gray_value', 'mean_value_R', 'mean_value_G', 'mean_value_B', 'std']]
# y = data['mean_gray_value']

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Crear el modelo de regresión lineal
# model = LinearRegression()

# # Entrenar el modelo con los datos de entrenamiento
# model.fit(X_train, y_train)

# # Hacer predicciones con el conjunto de prueba
# y_pred = model.predict(X_test)

# # Evaluar el rendimiento del modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # Mostrar los coeficientes del modelo
# print('Coefficients:', model.coef_)
# print('Intercept:', model.intercept_)

