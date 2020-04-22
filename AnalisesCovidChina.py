import pandas as pd
from matplotlib import pyplot
import numpy
from statsmodels.tsa.ar_model import AR
from datetime import timedelta 
import warnings
from datetime import datetime




NumeroDePrevisoes = 10



warnings.simplefilter(action='ignore', category=FutureWarning)

dados = pd.read_csv('time_series_covid19_confirmed_global-22-04.csv')
totalDeCasosPorDia = pd.DataFrame(columns = ['Data' , 'Casos'])
dados = dados.loc[dados['Country/Region']=='China']
data = dados.loc[:,'1/22/20':'4/21/20']


for i in range(len(data.columns)):
    dataAux = data.columns[i]+'20'
    dataIndex = datetime.strptime(dataAux, '%m/%d/%Y').date()
    soma = sum(data[data.columns[i]])
    totalDeCasosPorDia = totalDeCasosPorDia.append({'Data' :  dataIndex, 'Casos' : soma} , ignore_index=True)

totalDeCasosPorDia.to_csv('CasosDoMundo.csv')
Base = pd.read_csv('CasosDoMundo.csv', index_col='Data')
Base.drop('Unnamed: 0', inplace=True, axis=1)


dataFinal = Base

tamanho = dataFinal.size
ultimaData = totalDeCasosPorDia.iloc[tamanho-1].Data





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
numpy.save('man_model_mundo.npy', coef)
# Salva logs
lag = X[-window_size:]
numpy.save('man_data_mundo.npy', lag)
# Valva o ultimo ob
numpy.save('man_obs_mundo.npy', [series.values[-1]])




coef = numpy.load('man_model_mundo.npy')
print(coef)
lag = numpy.load('man_data_mundo.npy')
print(lag)
last_ob = numpy.load('man_obs_mundo.npy')
print(last_ob)

# carregua o modelo de AR do arquivo e faz uma previsão em uma etapa

 
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat


for j in range(NumeroDePrevisoes):
    # Carrega Modelo
    coef = numpy.load('man_model_mundo.npy')
    lag = numpy.load('man_data_mundo.npy')
    last_ob = numpy.load('man_obs_mundo.npy')
    # Faz Predição
    prediction = predict(coef, lag)
    # Tranforma a predição
    yhat = prediction + last_ob[0]
    print('Prediction: %f' % yhat)

    
    # Pega a opservação prevista
    observation = int(round(yhat[0]))
    # Atualiza e salva a nova observação
    lag = numpy.load('man_data_mundo.npy')
    last_ob = numpy.load('man_obs_mundo.npy')
    diffed = observation - last_ob[0]
    lag = numpy.append(lag[1:], diffed, axis=0)
    numpy.save('man_data_mundo.npy', lag)
    last_ob[0] = observation
    numpy.save('man_obs_mundo.npy', last_ob)
    
     
    
    ultimaData = ultimaData + timedelta(days=1)
    
    Dado = {'Casos' : yhat[0]}
    dataFinal = dataFinal.append(pd.DataFrame(Dado,index=[ultimaData]))
    
Base.plot(color='red')
dataFinal.plot()
pyplot.show()