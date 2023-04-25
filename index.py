from sys import argv

import numpy as np
import matplotlib as plt

import time
import csv
import re
import math

from scipy.signal import welch

from copy import deepcopy as dc

from filtros import *


delta = (0, 4)
theta = (4, 8)   # meditação, imaginação e criatividade
alpha = (8, 12)  # relaxamento e alerta, mas não focados em algo; calma, criatividade e meditação
beta = (12, 30)  # alerta e foco em atividade específica
gamma = (30, 100)


def carregamento():
    
    print("Fazendo o carregamento dos dados")
    
    with open('EEG_Rhythms/datasets/OpenBCI_GUI-v5-meditation.txt') as arquivo:
        linhas = arquivo.readlines()

    data = list()
    for i, linha in enumerate(linhas):
        res = re.search('^\d{1,3},((\ -?.+?,){8})', linha)
        if res:
            cols = res.group(1)
            data.append([float(d[1:]) for d in cols.split(',') if d])
    
    data = np.array(data[1:])
    
    np.save("EEG_Rhythms/datasets/OpenBCI_GUI-v5-meditation.npy", data)
    

def preprocessamento():

    print("Fazendo o preprocessamento dos dados")
    
    data = np.load('EEG_Rhythms/datasets/OpenBCI_GUI-v5-meditation.npy')
    X = data.swapaxes(1, 0)
    
    X_f = dc(X[:,7500:-3000])
    X_f = butter_notch(X_f, 60)
    X_f = butter_notch(X_f, 120)
    X_f = butter_lowpass(X_f, 50.)
    X_f = butter_highpass(X_f, 3.)
    
    np.save("EEG_Rhythms/datasets/OpenBCI_GUI-v5-meditation_filtered.npy", X_f)


def experimento(arquivo, taxa_amostragem, tempo, escala=False, simulacao=False):
    
    buffer = []

    # carrega os dados preparados e preprocessados
    data = np.load(arquivo)

    # define os campos que serão exibidos
    fields = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 

    # abre o arquivo para escrita
    with open(arquivo, 'w') as arquivo_csv:  
        escritor_csv = csv.writer(arquivo_csv)  
        escritor_csv.writerow(fields)

        buffer = data[:,0:20000]

        # verificar depois o tamanho do nperseg
        f, Pxx = welch(buffer, nperseg=128)

        X = np.average(Pxx, axis=0)

        features = list()

        for mi, ma in [delta, theta, alpha, beta, gamma]:
            features.append(X[mi:ma])

        features = [np.average(f) for f in features]

        # caso escala seja true, aplicar a escala
        if escala:
            buffer *= escala

        # caso seja uma simulação, exibir os prints
        if simulacao:
            print(f'[delta: {features[0]}, theta: {features[1]}, alpha: {features[2]}, beta:  {features[3]}, gamma: {features[4]}]')
            
        escritor_csv.writerow(features)

        # eu aguardo o tempo
        time.sleep(tempo)
       

def main():
    
    arquivo = argv[1]
    
    taxa_amostragem = float(argv[2]) 

    tempo = float(argv[3]) 
    
    if argv[4] == "-":
        escala = False
    else:
        escala = float(argv[4]) 
    
    if argv[5] == "-":    
        simulacao = False
    else:
        simulacao = bool(argv[5]) 
    
    carregamento()
    preprocessamento()
    experimento(arquivo, taxa_amostragem, tempo, escala, simulacao)

if __name__ == "__main__":
    main()






















        # 5 tempo - 250 é igual a taxa_amostragem, a quantidade de pontos que eu pego é de 1250

        # eu preciso definir o tamanho da janela
