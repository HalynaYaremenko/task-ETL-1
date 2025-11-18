import pandas as pd
import numpy as np

url = "https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/uk-500.csv"
df = pd.read_csv(url)

# 1. Імпорт та первинне дослідження

# print(df.head())
# print(df.info())
# print(df.describe())

# Перевірити якість:

# print(df.isna().sum()) # пропущені
# print(df.duplicated().sum()) # дублікати 

# 2. Очищення даних

# Видалити непотрібні колонки (за рішенням аналітика).

# Привести email та web до нижнього регістру.

df['email'] = df['email'].str.lower()
df['web'] = df['web'].str.lower()
# print(df.head())

# Очистити phone та fax від пробілів/символів.

df['phone'] = df['phone1'].str.strip()
df['fax'] = df['phone2'].str.strip()

# Стандартизувати формат текстових полів.

# 3. Створення нових колонок (Feature Engineering)
# Нова колонка	Опис
# full_name	Ім’я + прізвище
# email_domain	Домен email
# city_length	Довжина назви міста
# is_gmail	Boolean: чи email з gmail.com
