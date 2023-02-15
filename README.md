# Machine Learning Project

Tal y como muestran muchos titulares, en España ha aumentado notablemente el consumo de medicamentos ansiolíticos en la población.
Imaginemos que trabajamos como data scientist en una farmacéutica y queremos predecir el consumo del próximo año con el objetivo de producir una mayor o menor cantidad en función de la predicción de la demanda.

Para este proyecto se ha utilizado un dataset proveniente del Ministerio de Sanidad que recoge los datos del consumo de ansiolíticos de forma mensual desde 2010 hasta 2021. 

Al tratarse de datos de tipo time series, se ha analizado la estacionalidad de los datos así como su tendencia, que confirma que es de carácter creciente.

Después, se ha entrenado un modelo supervisado autoregresivo de tipo SARIMAX, consiguiendo una métrica de error MAE de 0.04.

Respecto a las predicciones, al utilizar como test los últimos datos de los que disponemos, que son los más peculiares ya que se ha producido un aumento drástico del consumo de ansiolíticos desde 2020 sin precedentes, debido a la situación de la pandemia así como a factores sociales y económicos, el modelo no logra predecir de forma óptima estos datos, ya que se alejan de la tendencia anterior.

Sin embargo, al haberlo entrenado después con todo el conjunto de datos, incluyendo los de los años 2020 y 2021, creemos que el modelo se puede comportar mejor en cuanto a las predicciones futuras.
