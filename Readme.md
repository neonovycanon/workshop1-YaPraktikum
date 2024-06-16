# *Startups operations/close predictions [M1_31DS+]*
### Проект мастерской Яндекс.Практикума

**Студент:** Георгий Косенко.
**Контакты:** t.me/neonocanon и gosha1411@gmail.com
#### Overview

Практико-ориентированное соревнование по курсу Специалист по Data Science плюс. 

**Задача:** Анализ *псевдо-реальных* данных о стартапах, функционировавших в период с 1970 по 2018 годы, с целью построения модели классификации, отвечающей на вопрос: *Закроется ли стартап?*

**Цели проекта:**
- Анализ данных, включающий:
    - Ознакомление с данными;
    - Предварительную обработку данных;
    - Полноценный `EDA` (Exploratory Data Analysis);
    - Проверку данных на *мультиколлинеарность*;
    - Выбор *набора обучающих признаков*.
- Выбор и описание наиболее подходящих моделей Машинного обучения;
- Обучение и тестирование моделей;
- Получение прогнозов модели;
- Анализ качества выбранной модели;
- Обзор важности признаков;
- Формирование итогового отчёта по исследованию.

**Результаты проекта :**
- Создан алгоритма сжатия многозначного категориального признака с использованием эмбеддиногов модели `glove-wiki-gigaword-300` на основе `Word2Vec`;
- Оценены и выбраны наиболее релевантные признаки, созданы синтетические признаки с более высокими показателями коэффициентов корреляции;
- Исследованы возможности моделей sklearn и Catboost, выбраны модели на основе `Support Vector Classifier`, `Catboost`;
- Получено значение тестовой метрики $\F_1$ равное 0.91191;
- В ходе исследования обнаружено качественное положительное влияние, связанное с исходными данными, подаваемыми на вход алгоритму сжатия категорий.