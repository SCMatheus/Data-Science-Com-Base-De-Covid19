# Ferramentas Estatísticas
# Síntaxe de importação e Apelidos
import numpy as np
import pandas as pd
import statistics as st
import scipy.stats as sp
import scipy.signal as ss
import statsmodels.graphics.gofplots as qq
import seaborn as sns


#Ferramentas para Análise Gráfica
import seaborn as sns
import matplotlib.pyplot as plt

#Outros
import warnings
import random as rd
warnings.simplefilter(action='ignore', category=FutureWarning)

dados = pd.read_csv('time_series_covid19_confirmed_global.csv')

dadosPorPais = dados.set_index('Country/Region')

dadosBrasil = pd.DataFrame(dadosPorPais.loc['Brazil', '1/22/20':'4/12/20']);
filtro  = dadosBrasil > 0
DadosBrasilDesdeInicioDosCasos = dadosBrasil[filtro]
DadosBrasilDesdeInicioDosCasos = DadosBrasilDesdeInicioDosCasos.dropna()

index = DadosBrasilDesdeInicioDosCasos.index
DadosBrasilDesdeInicioDosCasos.insert(0, "Data", index, True)

DadosBrasilDesdeInicioDosCasos = DadosBrasilDesdeInicioDosCasos.reset_index(drop=True)

DadosBrasilDesdeInicioDosCasos.rename(columns={'index': 'Data', 'Brazil': 'Casos'}, inplace = True)

for i in DadosBrasilDesdeInicioDosCasos.index:
    tamanho = len(DadosBrasilDesdeInicioDosCasos.iloc[i].Data)
    DadosBrasilDesdeInicioDosCasos.iloc[i].Data = DadosBrasilDesdeInicioDosCasos.iloc[i].Data[:(tamanho-3)];
    
plt.figure(figsize=(23,4))
ax = sns.barplot(x="Data", y="Casos", data = DadosBrasilDesdeInicioDosCasos)


