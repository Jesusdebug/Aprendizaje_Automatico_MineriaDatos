#Ciencia de datos en el conjunto de datos de alquiler de bicicletas
#Procesar datos, analizar y visualizar, encontrar insights, aplicar técnicas
#predictivas y razonar al respecto.

import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy import mean
from numpy import std
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
from pylab import rcParams
import seaborn as sns
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer

#Parte 1: Carga de datos

#Cargue los datos del repositorio UCI y colóquelos en la misma carpeta
#que el cuaderno.
#El enlace es https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset.

#Lea los datos del archivo .csv usando pandas.

hour_basis = pd.read_csv("bike+sharing+dataset/hour.csv")
day_basis = pd.read_csv("bike+sharing+dataset/day.csv")

#type cast datetime columns to date que tenga la mejor estructura
hour_basis["dteday"] = pd.to_datetime(hour_basis["dteday"])
day_basis["dteday"] = pd.to_datetime(day_basis["dteday"])

#Divida los datos de hour_basis en dos partes, una parte para los datos de reserva y el resto para el entrenamiento.

hold_out_hours_basis = hour_basis.loc[hour_basis['dteday'] >= "02/12/2012"]
hours_basis = hour_basis.loc[hour_basis['dteday'] < "02/12/2012"]

#Elimine la columna instantánea ya que es solo el número de registro

day_basis = day_basis.drop(columns=["instant"])
hours_basis = hours_basis.drop(columns=["instant"])
hold_out_hours_basis = hold_out_hours_basis.drop(columns=["instant"])

#Compruebe si faltan valores en los datos. No hay ningún valor faltante presente

hours_basis.isnull().sum(axis=0)

#Parte 2: Analizar y visualizar

#Visualizando Alquiler de Bicicletas por día.

#Se puede observar que el alquiler de bicicletas en 2012 es mayor en comparación con 2011.ared to 2011.

rcParams['figure.figsize'] = 25, 5
plt.plot(day_basis["cnt"])
plt.show()

#Visualizando el uso de la bicicleta por mes

#Se puede observar que la gente tiende a utilizar la bicicleta en
#verano/tiempo despejado en comparación con el tiempo frío.

rcParams['figure.figsize'] = 15, 5
sns.boxplot(x='mnth',y='cnt',data=day_basis)
plt.title("Número de Bicicletas prestada por mes")
plt.xlabel("Mes")
plt.ylabel("Cantidad")
plt.show()

#Visualización del uso de bicicleta por hora.

#El pico en el número de bicicletas se puede observar al principio y
#al final de las horas de trabajo, ya que la mayoría de las personas
#se desplazan a las oficinas en bicicleta. Sin embargo, el número de
#bicicletas alquiladas disminuye por la noche debido al frío.

rcParams['figure.figsize'] = 15,5
sns.boxplot(x='hr',y='cnt',data=hours_basis)
plt.title("Número de bicicletas prestadas por hora")
plt.xlabel("Hora")
plt.ylabel("Cantidad")
plt.show()

#Visualizando el impacto de la temperatura en el alquiler de bicicletas

rcParams['figure.figsize'] = 25,5
sns.boxplot(x='temp',y='cnt',data=hours_basis)
plt.title("Temperatura vs Prestamo de Bicicletas")
plt.xlabel("Hora")
plt.ylabel("Cantidad")
plt.show()

#Del gráfico anterior se desprende claramente que el alquiler de bicicletas
#aumenta en climas cálidos. Sin embargo, el alquiler de bicicletas disminuye
#si hace demasiado calor, es decir, por encima de 0,8.

#Visualización de la velocidad del viento frente al recuento de bicicletas en alquiler

rcParams['figure.figsize'] = 10,5
plt.scatter(hours_basis["windspeed"], hours_basis["cnt"], color='orange')
plt.title("Velocidad Viento Vs Prestamo Bicicletas")
plt.xlabel("Velocidad Viento")
plt.ylabel("Cantidad")
plt.show()

#Como era de esperar, la mayor velocidad del viento no es una condición favorable para el alquiler de bicicletas.

#Por el contrario, una menor velocidad del viento provoca un clima más cálido y agradable y, por tanto, un mayor alquiler de bicicletas.

#Visualización de la situación meteorológica frente al alquiler de bicicletas.

rcParams['figure.figsize'] = 10,5
sns.boxplot(x='weathersit',y='cnt',data=day_basis)
plt.title("Clima vs Alquiler de bicicletas")
plt.xlabel("Clima")
plt.xticks(np.arange(3), ('1:Despejado', '2:Niebla + Nublado', '3:Lluvia ligera/Nieve'))
plt.ylabel("Cantidad")
plt.show()

#Visualizando la temporada versus el alquiler de bicicletas
rcParams['figure.figsize'] = 10, 5
sns.boxplot(x = 'season', y = 'cnt', data = day_basis) 
plt.title("Estación vr Alquiler")
plt.xlabel("Estación")
plt.xticks(np.arange(4), ("1:invierno", "2:primavera", "3:verano", "4:otoño"))
plt.ylabel("Cantidad de Alquiler")
plt.show()

                                         
#1. Se puede observar que la variable objetivo está sesgada a la derecha. En la mayoría de los problemas de aprendizaje automático, es deseable la distribución gaussiana de los datos.
#2. Dado que se utilizaría Random Forest para predecir la demanda de datos de esta serie temporal, es importante comprobar la estacionariedad, es decir, la tendencia y la estacionalidad de los datos.
#3. Como se ve en la exploración de datos, contiene tanto tendencias como estacionalidad. Una de las formas de abordar este problema es agregar variables de retraso, diferenciación, transformación logarítmica o transformación box-cox.
#4. Elegí probar primero con variables de retraso, ya que se ve que la demanda de las horas anteriores tiene un gran impacto en la demanda de la hora actual.

#Comprobación de multicolinealidad

hours_basis.corr(method ='pearson')

#Según la matriz de correlación, temp y atemp están altamente correlacionados.
#temp muestra una relación más lineal con la variable objetivo, es decir, recuentos de alquiler en comparación con las características atemp.
#casual y registrado están altamente correlacionados con la columna cnt. Por lo tanto, ambas columnas se pueden eliminar.

#Parte 3: Construcción de modelos

#Agregar variables de retraso al conjunto de datos.

hours_basis["cnt_lag_1"] = hours_basis["cnt"].shift(-1)
hours_basis["cnt_lag_2"] = hours_basis["cnt"].shift(-2)
hours_basis = hours_basis.dropna()

#Dividir los datos en tren y prueba. Dado que se trata de datos de series de tiempo, los datos se dividen de forma secuencial en lugar de aleatoria.

testdata = hours_basis[hours_basis["dteday"] > "01/11/2012"].reset_index(drop=True) 
traindata = hours_basis[hours_basis["dteday"] <= "01/11/2012"].reset_index(drop=True)

#Una variable categórica de codificación activa como temporada, tiempo, mes, día laborable

testdata = pd.get_dummies(testdata, columns = ['season','weathersit','mnth','weekday'],drop_first=True)
traindata = pd.get_dummies(traindata, columns = ['season','weathersit','mnth','weekday'],drop_first=True)
hold_out_hours_basis = pd.get_dummies(hold_out_hours_basis,columns=['season','weathersit','mnth','weekday'],drop_first=True)

#Elimine columnas con alta multicolinealidad y columnas innecesarias del conjunto de datos.

target = ["cnt"]
features = list(traindata.columns)
features.remove("dteday")
features.remove("cnt")
features.remove("casual")
features.remove("registered")
features.remove("atemp")
Xtrain = traindata[features]
ytrain = traindata[target]
#Las columnas no están presentes en la prueba y mantienen el conjunto de datos
for col in features:
 if col not in testdata.columns:
     testdata[col] = 0
 if col not in hold_out_hours_basis.columns:
     hold_out_hours_basis[col] = 0

#Construya el conjunto de datos para el entrenamiento

Xtest = testdata[features]
ytest = testdata[target]
Xholdout = hold_out_hours_basis[features]
yholdout = hold_out_hours_basis[target]


#Elección de la métrica de evaluación

#RMSLE: error de registro de media cuadrática

#Mide el error relativo entre el valor real y el previsto. Tiene una alta tolerancia a las predicciones atípicas.
#Además, penaliza más las subestimaciones que las sobreestimaciones.
#En este caso, subestimar la demanda es más costoso que sobreestimarla, ya que puede haber pérdidas de negocio debido a la falta de disponibilidad de bicicletas.
#Junto con RMSLE, también se calcularían las puntuaciones RMSE, MAE y R2.#
#Entrene los datos utilizando Random Forest Regresor.

rf = RandomForestRegressor(max_depth=5, n_estimators = 6000, random_state=42)
rf.fit(Xtrain, ytrain)

