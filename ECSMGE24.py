#!/usr/bin/env python
# coding: utf-8

# CODE WITH FUNCTION TO PROCESS A FOLDER WITH MULTIPLE IMAGES

# In[48]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd

# Directory of images
diretorio_das_imagens = '...\\images\\'

# Function to calculate the area of the smallest circle circumscribed around a particle
def calcular_area_circunscrita(mascara):
    cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    if not cnts:
        return 0  # Retorna 0 se não houver contornos

    c = max(cnts, key=cv2.contourArea)

    # Find the smallest circumscribed circle
    ((_, _), raio) = cv2.minEnclosingCircle(c)

    # Calculate the area of the circle
    area_circunscrita = np.pi * (raio ** 2)

    return area_circunscrita

# Function to calculate areas of particles
def calcular_area_particulas(labels, otsu):
    resultados = []
    for label in np.unique(labels):
        if label == 0:
            continue
        mascara = np.zeros(otsu.shape, dtype='uint8')
        mascara[labels == label] = 255
        cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        area_circunscrita = calcular_area_circunscrita(mascara)
        resultados.append((label, area, area_circunscrita))
    return resultados

# Main function
def funcao_esfericidade(diretorio_das_imagens):
    dados_resultados = []

    # List all files in the directory
    arquivos_imagens = os.listdir(diretorio_das_imagens)

    # Iterate over the files and process each image
    for arquivo in arquivos_imagens:
        # Verifica se o arquivo é uma imagem (pode adicionar mais extensões se necessário)
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Constrói o caminho completo para a imagem
            caminho_completo = os.path.join(diretorio_das_imagens, arquivo)

            # Open the image
            imagem = Image.open(caminho_completo)

            # Convert the image to a numpy array for visualization and processing
            imagem_array = np.array(imagem)

            # Invert the colors
            inv_imagem_array = 255 - imagem_array

            # Gaussian blur
            desfoque = cv2.GaussianBlur(inv_imagem_array, (5, 5), 0)

            # Otsu's method - global threshold, i.e., over the entire image, automatically
            valor, otsu = cv2.threshold(desfoque, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Plotting the distance matrix
            dist = ndi.distance_transform_edt(otsu)

            # Locating local maxima in the distance matrix
            local_max = peak_local_max(dist, indices=False, min_distance=20, labels=otsu)

            # Labeling local maxima
            markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]

            # Applying watershed
            labels = watershed(-dist, markers, mask=otsu)

            # Individualizing particles
            img_final = labels.copy()
            for label in np.unique(labels):
                if label == 0:
                    continue
                mascara = np.zeros(otsu.shape, dtype='uint8')
                mascara[labels == label] = 255
                cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
                c = max(cnts, key=cv2.contourArea)
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.putText(img_final, "{}".format(label), (int(x) - 1, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 200), 2)

            # Calculate particle areas
            resultados_areas = calcular_area_particulas(labels, otsu)

            # Add results to the overall list
            for label, area, area_circunscrita in resultados_areas:
                dados_resultados.append((arquivo, label, area, area_circunscrita))

    # Convert the list of data to a pandas DataFrame
    df_resultados = pd.DataFrame(dados_resultados, columns=['Nome do Arquivo', 'Número da Partícula', 'Área', 'Área Circunscrita'])

    # Specify the Excel file saving path
    caminho_excel = '...\\Table.xlsx'

    # Export the DataFrame to an Excel file
    df_resultados.to_excel(caminho_excel, index=False)
    print(f"O resultado foi salvo em: {caminho_excel}")

# Example of using the function
funcao_esfericidade(diretorio_das_imagens)


# In[ ]:




