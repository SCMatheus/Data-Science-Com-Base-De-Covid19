# Ferramentas Estatísticas
# Síntaxe de importação e Apelidos

import pandas as pd
from matplotlib import pyplot
import numpy
from statsmodels.tsa.ar_model import AR
from datetime import timedelta 
import warnings




NumeroDePrevisoes = 10



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




#Salva Nova Base 
ano = '2020'
for i in DadosBrasilDesdeInicioDosCasos.index:
    tamanho = len(DadosBrasilDesdeInicioDosCasos.iloc[i].Data)
    aux = DadosBrasilDesdeInicioDosCasos.iloc[i].Data[:(tamanho-2)];
    DadosBrasilDesdeInicioDosCasos.iloc[i].Data =  aux + ano

DadosBrasilDesdeInicioDosCasos.to_csv('FiltradoPorBrasil.csv')

#Carrega Nova Base
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y') 
Base = pd.read_csv('FiltradoPorBrasil.csv', parse_dates=['Data'], index_col='Data',date_parser=dateparse)
Base.drop('Unnamed: 0', inplace=True, axis=1)


dataFinal = Base

tamanho = dataFinal.size
ultimaData = pd.datetime.strptime(DadosBrasilDesdeInicioDosCasos.iloc[tamanho-1].Data, '%m/%d/%Y')





# cria uma transformação de diferença do conjunto de dados
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
 
# Carrega Base de Dados
series = dataFinal
X = difference(series.Casos)
# Treina o modelo
window_size = 6
model = AR(X)
model_fit = model.fit(maxlag=window_size, disp=False)
# Salva os coeficientes
coef = model_fit.params
numpy.save('man_model.npy', coef)
# Salva logs
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
# Valva o ultimo ob
numpy.save('man_obs.npy', [series.values[-1]])




coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)

# carregua o modelo de AR do arquivo e faz uma previsão em uma etapa

 
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat


for j in range(NumeroDePrevisoes):
    # Carrega Modelo
    coef = numpy.load('man_model.npy')
    lag = numpy.load('man_data.npy')
    last_ob = numpy.load('man_obs.npy')
    # Faz Predição
    prediction = predict(coef, lag)
    # Tranforma a predição
    yhat = prediction + last_ob[0]
    print('Prediction: %f' % yhat)

    
    # Pega a opservação prevista
    observation = int(round(yhat[0]))
    # Atualiza e salva a nova observação
    lag = numpy.load('man_data.npy')
    last_ob = numpy.load('man_obs.npy')
    diffed = observation - last_ob[0]
    lag = numpy.append(lag[1:], diffed, axis=0)
    numpy.save('man_data.npy', lag)
    last_ob[0] = observation
    numpy.save('man_obs.npy', last_ob)
    
     
    
    ultimaData = ultimaData + timedelta(days=1)
    
    Dado = {'Casos' : yhat[0]}
    dataFinal = dataFinal.append(pd.DataFrame(Dado,index=[ultimaData]))
    
Base.plot(color='red')
dataFinal.plot()
pyplot.show()