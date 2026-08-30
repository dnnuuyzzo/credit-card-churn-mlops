#!/bin/bash
REST_API_PORT="${PORT:-8501}"
echo "Starting TensorFlow Serving on REST API port ${REST_API_PORT} with Prometheus monitoring..."
exec tensorflow_model_server --port=8500 --rest_api_port="${REST_API_PORT}" \
  --model_name="${MODEL_NAME:-credit_card_churn_model}" \
  --model_base_path="/models/${MODEL_NAME:-credit_card_churn_model}" \
  --monitoring_config_file="${MONITORING_CONFIG:-/model_config/prometheus.config}"