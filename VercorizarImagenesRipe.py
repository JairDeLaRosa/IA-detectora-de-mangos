# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:19:26 2024

@author: HP
"""
        
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import statistics
import pandas as pd

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
    #plt.imshow(mango_mask)
    #plt.show() 

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
    
    print(df_image.head())
    
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
    plt.imshow(masked_gray, cmap = 'gray')
    plt.show()
    df_intensities = intensity_table(masked_gray)
    print(df_intensities)
    return df_intensities

img_file1 = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/Ripe/166.jpg"
img_file2 = "C:/Users/HP/OneDrive - Universidad del Magdalena/Documentos/archive/dataset/train/Rotten/6.jpg."

grayImg1, img1 = convert_toGrayImage(img_file1)
grayImg2, img2 = convert_toGrayImage(img_file2)

mango_mask1 = get_binaryMango(grayImg1)
mango_mask2 = get_binaryMango(grayImg2)

#Dos opciones para vectorizar las imagenes: podría tomar una de las dos opciones

#  --> opción 1, Cada valor de la escala de grises se toma como una característica de la imagen.
hist2features(grayImg1, mango_mask1)
hist2features(grayImg2, mango_mask2)

#  --> opción 2, se extraen características de las imagenes utilizando la función regionprops, aquí se extraen algunas propiedades; pero
#      podrían tomarse más características.
extract_properties(img1, grayImg1, mango_mask1)
extract_properties(img2, grayImg2, mango_mask2)