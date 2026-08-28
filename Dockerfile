FROM tensorflow/serving:2.11.0

COPY serving_model_dir/credit_card_churn_model /models/credit_card_churn_model
ENV MODEL_NAME=credit_card_churn_model

COPY run.sh /usr/bin/run.sh
RUN chmod +x /usr/bin/run.sh

EXPOSE 8500
EXPOSE 8501

ENTRYPOINT ["/bin/bash", "/usr/bin/run.sh"]
