# Ferramentas Estatísticas
# Síntaxe de importação e Apelidos

import pandas as pd
import seaborn as sns
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import numpy
from pandas import read_csv
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import ARResults

#Ferramentas para Análise Gráfica
import seaborn as sns
import matplotlib.pyplot as plt
#Outros
import warnings


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

'''
for i in DadosBrasilDesdeInicioDosCasos.index:
    tamanho = len(DadosBrasilDesdeInicioDosCasos.iloc[i].Data)
    DadosBrasilDesdeInicioDosCasos.iloc[i].Data = DadosBrasilDesdeInicioDosCasos.iloc[i].Data[:(tamanho-3)];
    
plt.figure(figsize=(23,4))
ax = sns.barplot(x="Data", y="Casos", data = DadosBrasilDesdeInicioDosCasos)
'''



#Salva Nova Base 
ano = '2020'
for i in DadosBrasilDesdeInicioDosCasos.index:
    tamanho = len(DadosBrasilDesdeInicioDosCasos.iloc[i].Data)
    aux = DadosBrasilDesdeInicioDosCasos.iloc[i].Data[:(tamanho-2)];
    DadosBrasilDesdeInicioDosCasos.iloc[i].Data =  aux + ano

DadosBrasilDesdeInicioDosCasos.to_csv('FiltradoPorBrasil.csv')

#Carrega Nova Base
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y') 
data = pd.read_csv('FiltradoPorBrasil.csv', parse_dates=['Data'], index_col='Data',date_parser=dateparse)
data.drop('Unnamed: 0', inplace=True, axis=1)

dataFinal = data 

'''
plt.plot(ts)
'''
'''
#exibe medias moveis
test_stationarity(data)


rolling = data.rolling(window=3)
rolling_mean = rolling.mean()
print(rolling_mean.head(60))
# plot original and transformed dataset
data.plot()
rolling_mean.plot(color='red')
pyplot.show()


series = data
X = series.Casos
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
	length = len(history)
	yhat = mean([history[i] for i in range(length-window,length)])
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
# zoom plot
pyplot.plot(test[0:100])
pyplot.plot(predictions[0:100], color='red')
pyplot.show()
'''



 #Modelo de previsão de séries temporais

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
 
# Carrega dataset
series = dataFinal
X = difference(series.Casos)
# fit model
model = AR(X)
model_fit = model.fit(maxlag=6, disp=False)
# Salva modelo em arquivo
model_fit.save('ar_model.pkl')
#salva o conjunto de dados diferenciado
numpy.save('ar_data.npy', X)
# salva o último ob
numpy.save('ar_obs.npy', [series.values[-1]])




#carrega o modelo de AR do arquivo
loaded = ARResults.load('ar_model.pkl')
print(loaded.params)
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
print(last_ob)






'''
# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)
 
# load dataset
series = dataFinal
X = difference(series.Casos)
# fit model
window_size = 6
model = AR(X)
model_fit = model.fit(maxlag=window_size, disp=False)
# save coefficients
coef = model_fit.params
numpy.save('man_model.npy', coef)
# save lag
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
# save the last ob
numpy.save('man_obs.npy', [series.values[-1]])




coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)

'''
# carregua o modelo de AR do arquivo e faz uma previsão em uma etapa
# Carrega o modelo
model = ARResults.load('ar_model.pkl')
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
# Faz a predição
predictions = model.predict(start=len(data), end=len(data))
# transforma a previsão
yhat = predictions[0] + last_ob[0]
print('Prediction: %f' % yhat)

'''
# load AR model from file and make a one-step prediction

 
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat
 
# load model
coef = numpy.load('man_model.npy')
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
# make prediction
prediction = predict(coef, lag)
# transform prediction
yhat = prediction + last_ob[0]
print('Prediction: %f' % yhat)
'''
#atualiza os dados para o modelo AR com uma nova obs

# Pega a Observação
observation = 24613.904245
# Carrega o modelo salvo
data = numpy.load('ar_data.npy')
last_ob = numpy.load('ar_obs.npy')
#atualiza e salva observação diferenciada
diffed = observation - last_ob[0]
data = numpy.append(data, diffed, axis=0)
numpy.save('ar_data.npy', data)
# atualiza e salva a observação real
last_ob[0] = observation
numpy.save('ar_obs.npy', last_ob)


'''

# get real observation
observation = 24613.904245
# update and save differenced observation
lag = numpy.load('man_data.npy')
last_ob = numpy.load('man_obs.npy')
diffed = observation - last_ob[0]
lag = numpy.append(lag[1:], diffed, axis=0)
numpy.save('man_data.npy', lag)
# update and save real observation
last_ob[0] = observation
numpy.save('man_obs.npy', last_ob)
'''