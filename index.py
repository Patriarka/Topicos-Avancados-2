from sys import argv

import numpy as np

import time
import csv
import re

from scipy.signal import welch

from copy import deepcopy as dc

from filtros import *
from sklearn.preprocessing import minmax_scale


delta = (0, 4)
theta = (4, 8)   # meditação, imaginação e criatividade
alpha = (8, 12)  # relaxamento e alerta, mas não focados em algo; calma, criatividade e meditação
beta = (12, 30)  # alerta e foco em atividade específica
gamma = (30, 100)

data = None
X_f = None

def carregamento(arquivo):
    
    global data
    
    print("Fazendo o carregamento dos dados")
    
    with open(arquivo) as arquivo:
        linhas = arquivo.readlines()

    data = list()
    for i, linha in enumerate(linhas):
        res = re.search('^\d{1,3},((\ -?.+?,){8})', linha)
        if res:
            cols = res.group(1)
            data.append([float(d[1:]) for d in cols.split(',') if d])
    
    data = np.array(data[1:])
    

def preprocessamento(taxa_amostragem):
    
    global data
    global X_f

    print("Fazendo o preprocessamento dos dados")
    
    X = data.swapaxes(1, 0)
    
    X_f = dc(X[:,7500:-3000])
    X_f = butter_notch(X_f, 60, fs=taxa_amostragem)
    X_f = butter_notch(X_f, 120, fs=taxa_amostragem)
    X_f = butter_lowpass(X_f, 50., fs=taxa_amostragem)
    X_f = butter_highpass(X_f, 3., fs=taxa_amostragem)

def experimento(taxa_amostragem=250, tempo=1, escala=False, simulacao=False, update=1):
    
    global X_f
    
    buffer = []

    # define os campos que serão exibidos
    fields = ['max_momento', 'delta', 'theta', 'alpha', 'beta', 'gamma'] 
        
    # abre o arquivo para escrita
    with open('results.csv', 'w', newline='') as arquivo_csv:  
        escritor_csv = csv.writer(arquivo_csv)  
        escritor_csv.writerow(fields)

        # tamanho da janela para calcular a PSD com welch
        janela = int(taxa_amostragem * tempo)  

        # variável para armazenar o momento máximo
        max_momento = tempo
        taxa_att = 0
        
        while True:
            begin_index = int(taxa_att * taxa_amostragem)
            end_index = int(taxa_att * taxa_amostragem) + janela

            taxa_att += update

            # seleciona a janela atual
            buffer = X_f[:, begin_index : end_index]  
            
            # verifica se a janela atual é menor que a janela definida, se sim, interrompe o loop pois chegou ao fim
            if buffer.shape[1] < janela:
                break
            
            nperseg = 256
            
            if janela < 256:
                nperseg = min(janela, buffer.shape[1])

            # Calcula o espectrograma usando o método de Welch
            f, Pxx = welch(buffer, fs=taxa_amostragem, nperseg=nperseg)

            # Calcula a média das potências espectrais para cada janela
            X = np.average(Pxx, axis=0)

            # Cria uma lista vazia para armazenar as características de frequência
            features = list()

            # Define as faixas de frequência para cada tipo de onda cerebral
            # (delta, theta, alpha, beta e gamma)
            for mi, ma in [delta, theta, alpha, beta, gamma]:
                features.append(X[mi:ma])

            # Calcula a média das intensidades das ondas cerebrais para cada faixa de frequência
            features = [np.average(f) for f in features]

            escritor_csv.writerow([max_momento, *features])
            
            # caso escala tenha um valor, aplicar a escala
            if escala:
                features = minmax_scale(features, feature_range=(0, escala))
                
            # caso seja uma simulação, exibir os prints
            if simulacao:
                print(f'[momento: {max_momento}, delta: {features[0]}, theta: {features[1]}, alpha: {features[2]}, beta:  {features[3]}, gamma: {features[4]}]')
                time.sleep(update)    
            
            max_momento += update

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
    
    update = float(argv[6])

    carregamento(arquivo)
    
    preprocessamento(taxa_amostragem)
        
    experimento(taxa_amostragem, tempo, escala, simulacao, update)

if __name__ == "__main__":
    main()