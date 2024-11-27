FROM python:3.12.7-slim-bookworm

EXPOSE 8080
WORKDIR /app

COPY ./requirements.txt requirements.txt
COPY ./car_price_prediction_pipeline.pkl car_price_prediction_pipeline.pkl
COPY ./service.py service.py

RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["fastapi", "run", "service.py", "--port", "8080"]
