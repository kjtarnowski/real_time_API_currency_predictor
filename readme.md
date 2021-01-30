# Currency predictor

## Summary:
Application dedicated to perform periodic tasks:
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




