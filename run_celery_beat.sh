#!/bin/sh

# wait for RabbitMQ server to start
sleep 5


celery -A real_time_API_currency_predictor beat -l info


