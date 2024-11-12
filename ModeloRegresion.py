import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import statistics
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from skimage.data import page
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

def convert_toGrayImage(filename):
    img = ski.io.imread(filename)
    img = ski.util.img_as_ubyte(img)
    gray_image = rgb2gray(img)
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
    mango_mask = mango_mask < 1
    return mango_mask

def std_image(gray_img, mango_mask):
    mango_mask = ski.util.invert(mango_mask)
    idx_list = np.where(mango_mask == 1)
    mangoPixel_values = gray_img[idx_list]
    stdev = statistics.stdev(mangoPixel_values)
    return stdev

def extract_properties(mango_img, gray_img, mango_mask):

    labeled_mango, count = ski.measure.label(mango_mask, connectivity=2, return_num=True)
    object_features = ski.measure.regionprops(labeled_mango)

    var_color_props = ['intensity_mean']
    color_props = ski.measure.regionprops_table(labeled_mango, mango_img, properties=var_color_props)

    var_gris_props = ['area', 'intensity_max', 'intensity_min', 'intensity_mean']
    gris_props = ski.measure.regionprops_table(labeled_mango, gray_img, properties=var_gris_props)
    
    grayImg_stdev = std_image(gray_img, mango_mask)
    df_image = pd.DataFrame(gris_props)
    
    df_image.rename(columns={'intensity_max': 'max_gray_value', 'intensity_min': 'min_gray_value', 'intensity_mean': 'mean_gray_value'}, inplace=True)
    
    df_image['mean_value_R'] = color_props['intensity_mean-0'] 
    df_image['mean_value_G'] = color_props['intensity_mean-1'] 
    df_image['mean_value_B'] = color_props['intensity_mean-2']
    df_image['std'] = grayImg_stdev 
    
    return df_image

def process_images_in_folder(folder_path, label):
    properties_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            gray_img, img = convert_toGrayImage(file_path)
            mango_mask = get_binaryMango(gray_img)
            properties_df = extract_properties(img, gray_img, mango_mask)
            properties_df['label'] = label
            properties_list.append(properties_df)
    
    return pd.concat(properties_list)

# Paths to the folders containing the images
ripe_folder_path = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRipe"
rotten_folder_path = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/subRotten"

# Process images in both folders and combine the results
ripe_properties_df = process_images_in_folder(ripe_folder_path, "Ripe")
rotten_properties_df = process_images_in_folder(rotten_folder_path, "Rotten")

# Combine the dataframes from both folders
combined_properties_df = pd.concat([ripe_properties_df, rotten_properties_df])

# Save the combined dataframe to a CSV file
combined_properties_df.to_csv("mango_properties.csv", index=False)

print("The properties of the images have been successfully saved to mango_properties.csv.")

# Cargar los datos desde el archivo CSV
data = pd.read_csv('mango_properties.csv')

# Supongamos que tienes una columna 'label' que indica si el mango está maduro (1) o podrido (0)
# Seleccionar las variables independientes (X) y la variable dependiente (y)
X = data[['max_gray_value', 'min_gray_value', 'mean_value_R', 'mean_value_G', 'mean_value_B', 'std']]
y = data['label']  # Asegúrate de tener esta columna en tu DataFrame

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

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))