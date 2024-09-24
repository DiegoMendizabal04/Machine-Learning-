#1- Importamos las librerias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense 
from sklearn.preprocessing import MinMaxScaler
import requests 

#2- Configuramos la API utilizando Polygon
api_key='QRDMyy_zsH6nYwHwApvJ6tdiU88etVu2'
ticker= 'MCD'

#3- Creamos una funcion para obtener los datos de la API de Polygon 
#3.1- Los datos los pasamos a un archivo json y si la conexion es exitosa lo convertimos en un DataFrame
#3.2- En caso de que la conexión no sea exitosa le pedimos al sistema que nos avise y devuelva el df vacio 
def get_polygon_data(ticker, api_key):
    url=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/1994-01-01/2024-01-01?apiKey={api_key}"
    response=requests.get(url)

    if response.status_code== 200:
      data=response.json()
      if 'results' in data:
        return pd.DataFrame(data['results'])
      
      else:
        print("No se encontraron resultados para el símbolo proporcionado.")
        return pd.DataFrame()  # Devuelve un DataFrame vacío si no hay resultados

    else:
       print(f"Error al obtener los datos: {response.status_code}")
       return pd.DataFrame()  
#3.3 Cambiamos el formato de la fecha por un formato mas legible
df = get_polygon_data(ticker, api_key)
df['date'] = pd.to_datetime(df['t'], unit='ms')

#4-Cambiamos el nombre de las columnas para una mejor identificiacion
df.rename(columns={
   'v': 'Volumen',
    'vw': 'Precio_Promedio_Ponderado',
    'o': 'Precio_Apertura',
    'c': 'Precio_Cierre',
    'h': 'Precio_Maximo',
    'l': 'Precio_Minimo',
    't': 'Marca_Tiempo',
    'n': 'Numero_Transacciones',
    'date': 'Fecha'
}, inplace=True)

#5-Establecemos la columna 'Fecha' como indice y solamente utilizamos los precios de cierre de las acciones
df.set_index('Fecha', inplace=True)
prices=df[['Precio_Cierre']] #Solamente usamos los precios de cierre

#6- Escalado de datos 
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(prices)

#7-Creamos el conjunto de datos de entrenamiento y prueba
train_size= int(len(scaled_data)* 0.7)#Usamos el 70% de los datos para entrenamiento en lugar de 80% para obtener un numero valido de salidas de test_predict
test_size= len(scaled_data)-train_size
train_data, test_data= scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]
time_step=60 

#8-Funcion para crear el dataset con LSTM
def create_dataset(dataset, time_step=60): 
   X, Y=[], []
   for i in range(len(dataset)-time_step -1):  #el bucle itera desde el 0 hasta la longitud del dataset menos el timestep
       a=dataset[i:(i+ time_step), 0] #creamos las secuencias para X, 0 hace referencia a la primera fila. En este caso precios de cierre escalados 
       X.append(a)
       Y.append(dataset[i + time_step, 0]) #Es el valor que esta inmediatamente despues del 0, es decir el valor "61"
      
   return np.array(X), np.array(Y)  


#9-Creamos los conjuntos de datos con el time_step fijo 
X_train, y_train= create_dataset(train_data, time_step)
X_test, y_test= create_dataset(test_data, time_step)

#10-reshape de los datos para el modelo LSTM 
X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


#11- Construcción del modelo LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # Definimos la forma de entrada utilizando Input()
model.add(LSTM(50, return_sequences=True))    # Capa LSTM 1
model.add(LSTM(50, return_sequences=False))   # Capa LSTM 2
model.add(Dense(1))                           # Capa Densa

model.compile(optimizer='adam', loss='mean_squared_error')

#12-Entrenamos el modelo 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

#13-realizamos las predicciones de nuestro modelo 
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
#13.1 Realizamos la inversa del escalado para obtener los valores reales 
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

#14 preparamos los datos para la visualización 
train_plot= np.empty_like(scaled_data)
train_plot[:,:]=np.nan 
train_plot[time_step:len(train_predict)+ time_step,:]= train_predict


test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
start_idx = len(train_predict) + (time_step * 2)
test_plot[start_idx:start_idx + len(test_predict), :] = test_predict
""""
test_plot= np.empty_like(scaled_data)
test_plot[:, :]=np.nan 
test_plot[len(train_predict)+ (time_step*2)+1:len(scaled_data)-1,:]=test_predict
"""
# 15 - Graficamos y guardamos los datos
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Training Predictions')
plt.plot(test_plot, label='Test Predictions')
plt.title('Stock price predictions using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

#16' Guardamos el gráfico en un archivo PNG antes de mostrarlo
plt.savefig('stock_price_predictions.png', dpi=300, bbox_inches='tight') 
plt.show()
