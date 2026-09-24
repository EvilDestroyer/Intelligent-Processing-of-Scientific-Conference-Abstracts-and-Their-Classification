# Intelligent Processing of Scientific Conference Abstracts and Their Classification
Итоговая аттестационная работа бакалавра на тему «Применение методов машинного обучения для интеллектуальной обработки тезисов научных конференций и распределения их по классам». Работа выполнялась в учреждении высшего образования "Московский авиационный институт (национальный исследовательский университет)".

Все файлы кроме папки Theses должны находится в одной папке! Папка Theses и её содержимое может находится в любом месте на компьютере.
Запуск происходит через файл Interface.py!


## Файлы в проекте

**Interface.py** - запуск интерфейса и связь полей и кнопок с функциями

**PostgreSQL.py** - загрузка данных в базу данных PostgreSQL и их предобработка

**CatBoost_Classifier.py** - классификация обученной моделью градиентного бустинга CatBoost

**CatBoost_Train.py** - программа для тренировки и сохранения модели градиентного бустинга CatBoost

**Graphs.py** - вывод статистики и графиков распределения

**resources_rc.py** - загрузка изображений для интерфейса

**thesis_classifier_catboost.pkl** - сохранённая обученная модель градиентного бустинга CatBoost

**Program Interface.ui** - интерфейс программы

**Theses** - папка с тремя файлами: тезисы в формате docx, категории в формате txt и авторы в формате txt. Представлены в качестве примера как должны выглядеть загружаемые файлы. Все тезисы и категории были взяты из открытых источников конференции "24-я Международная конференция «Авиация и космонавтика»".


## Пример работы программы:
<img width="781" height="811" alt="изображение" src="https://github.com/user-attachments/assets/dd675296-a601-45d5-b28d-ceda65446799" />

<img width="285" height="126" alt="изображение" src="https://github.com/user-attachments/assets/ecb60dfd-3b3f-463e-abc8-2ecf91736fe5" />

<img width="1201" height="800" alt="изображение" src="https://github.com/user-attachments/assets/131ee1ea-01e6-4d28-ab5c-ff0ee5d0dc56" />

<img width="810" height="557" alt="изображение" src="https://github.com/user-attachments/assets/adc50086-d190-43b1-8335-9fdffc487e51" />

<img width="676" height="496" alt="изображение" src="https://github.com/user-attachments/assets/511d4b50-9967-44db-9eb9-ff05fc667a00" />
