FROM tensorflow/serving:2.11.0

COPY serving_model_dir/credit_card_churn_model /models/credit_card_churn_model

ENV MODEL_NAME=credit_card_churn_model

EXPOSE 8500
EXPOSE 8501

ENTRYPOINT []

CMD ["sh", "-c", "tensorflow_model_server --port=8500 --rest_api_port=$PORT --model_name=$MODEL_NAME --model_base_path=/models/$MODEL_NAME"]