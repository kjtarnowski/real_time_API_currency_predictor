# Currency predictor

## Summary:
Prototype application dedicated to perform periodic tasks:
- getting currency rates from web sources (web scraping or APIs) and storing in database
- optimizing deep learning time series models (GRU - Gated recurrent unit) by the use of
Ray Tune - hyperparameter tuning library
- fitting time series models to data stored in database
- predicting future currency rates values based deep learning models

Application allows retrieving data via the REST API interface. 

## Technologies/libraries
* Python
* Django and Django REST framework
* Tensorflow
* Scikit-learn
* Pandas
* Tensorflow
* Ray Tune
* Beautiful Soup
* Celery
* RabbitMQ
* Celery Beat scheduler

## Requirements
* docker and docker-compose must be installed on the system

## Instalation

```
docker-compose up -d --build
```

## Comments
Current prototype version of application takes currency rates 
(euro/dollar and pound/dollar) from  
https://www.investing.com/currencies/single-currency-crosses.
The steps of fitting, optimizing and predicting the model are preceded by
data collection stage (about 3.5 hours, celery web scraping task 
performed every minute). After data gathering, prediction is
executed every minute, fitting model is performed every 15 minutes,
and model optimization is executed about every hour. 




