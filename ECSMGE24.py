#!/usr/bin/env python
# coding: utf-8

# CÓDIGO COM FUNÇÃO PARA RODAR UMA PASTA COM VÁRIAS IMAGENS

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

# Diretório das imagens
diretorio_das_imagens = '...\\images\\'

# Função para calcular a área do menor círculo circunscrito a uma partícula
def calcular_area_circunscrita(mascara):
    cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    if not cnts:
        return 0  # Retorna 0 se não houver contornos

    c = max(cnts, key=cv2.contourArea)

    # Encontrar o menor círculo circunscrito
    ((_, _), raio) = cv2.minEnclosingCircle(c)

    # Calcular a área do círculo
    area_circunscrita = np.pi * (raio ** 2)

    return area_circunscrita

# Função para calcular áreas das partículas
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

# Função principal
def funcao_esfericidade(diretorio_das_imagens):
    dados_resultados = []

    # Lista todos os arquivos no diretório
    arquivos_imagens = os.listdir(diretorio_das_imagens)

    # Itera sobre os arquivos e processa cada imagem
    for arquivo in arquivos_imagens:
        # Verifica se o arquivo é uma imagem (pode adicionar mais extensões se necessário)
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Constrói o caminho completo para a imagem
            caminho_completo = os.path.join(diretorio_das_imagens, arquivo)

            # Abrindo a imagem
            imagem = Image.open(caminho_completo)

            # Conversão da imagem em "array numpy" para visualização e processamento
            imagem_array = np.array(imagem)

            # Inversão das cores
            inv_imagem_array = 255 - imagem_array

            # Desfoque gaussiano
            desfoque = cv2.GaussianBlur(inv_imagem_array, (5, 5), 0)

            # Método de Otsu - limiar global, ou seja, em toda a imagem, de forma automática
            valor, otsu = cv2.threshold(desfoque, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # Plotando a matriz de distância
            dist = ndi.distance_transform_edt(otsu)

            # Localizando os máximos locais na matriz de distância
            local_max = peak_local_max(dist, indices=False, min_distance=20, labels=otsu)

            # Rotulando os máximos locais
            markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]

            # Aplicando watershed
            labels = watershed(-dist, markers, mask=otsu)

            # Individualizando as partículas
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

            # Calcular áreas das partículas
            resultados_areas = calcular_area_particulas(labels, otsu)

            # Adicionar resultados à lista geral
            for label, area, area_circunscrita in resultados_areas:
                dados_resultados.append((arquivo, label, area, area_circunscrita))

    # Converter a lista de dados em um DataFrame do pandas
    df_resultados = pd.DataFrame(dados_resultados, columns=['Nome do Arquivo', 'Número da Partícula', 'Área', 'Área Circunscrita'])

    # Especifica o caminho de salvamento do arquivo Excel
    caminho_excel = '...\\Table.xlsx'

    # Exporta o DataFrame para um arquivo Excel
    df_resultados.to_excel(caminho_excel, index=False)
    print(f"O resultado foi salvo em: {caminho_excel}")

# Exemplo de uso da função
funcao_esfericidade(diretorio_das_imagens)


# In[ ]:




