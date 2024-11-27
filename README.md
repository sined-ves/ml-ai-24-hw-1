# ML AI-24 HW-1

### Содержание:
* [Ссылка на colab с аналитикой](https://colab.research.google.com/drive/1IllFvsYihyLuAfElc_Q2zU_O0tPFEb5h?usp=sharing)
* [Ноутбук с аналитикой](AI_HW1_Regression_with_inference_base_Denis_Bobylev.ipynb)
* [Отчеты YData Profiling](ydata-reports)
* [FastAPI Сервис](service.py)
* [.pkl файл пайплайна](car_price_prediction_pipeline.pkl)

### Отчет:
#### Что было сделано:
* проделан базовый EDA и выполнена предобработка данных
* построены отчеты в YData Profiling и выполнен анализ зависимостей в данных
* обучены модели регрессии для предсказания стоимости автомобилей:
  - наивная на вещественных признаках
  - наивная на стандартизированных вещественных признаках
  - Lasso на стандартизированных вещественных признаках
  - ElasticNet на стандартизированных вещественных признаках
  - Ridge на стандартизированных вещественных и закодированных OHE категориальных признаках
* реализован веб-сервис для применения построенной модели на новых данных
#### Что не было сделано:
* не обработан столбец torque
#### Общие выводы:
* Было интересно обучить разные виды моделей линейной регрессии и сравнить их результаты
* Показалось странным, что L1 регуляризация не занулила никакие веса 
* Значительно улучшить качество модели удалось, благодаря добавлению категориальных признаков, особенно name - это в целом ожидаемо, так как марка машины сильно влияет на ее стоимость

### Инференс
* как запустить
```
docker build -t sined-ves-ml-ai-24-hw-1:v1.0.0 .
docker run --rm -d -p 8080:8080 --name sined-ves-ml-ai-24-hw-1 sined-ves-ml-ai-24-hw-1:v1.0.0
```
* [screencast](https://drive.google.com/file/d/1BIfGQHDQuwEizoZ6rGK2H25HvfukRX6N/view?usp=drive_link)
