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

# def experimento(arquivo, taxa_amostragem, tempo, escala=False, simulacao=False):

    
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
        simulacao = bool(argv[5]) if len(argv) >= 6 else False
    
    carregamento()
    preprocessamento()
    # experimento(arquivo, taxa_amostragem, tempo, escala, simulacao)

if __name__ == "__main__":
    main()