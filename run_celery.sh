#!/bin/sh

# wait for RabbitMQ server to start
sleep 10

celery -A real_time_API_currency_predictor worker -l info


