import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 50) # макс к-ть стовпчиків, які ми будемо бачити у терміналі
pd.set_option("display.width", 180)

url = "https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/uk-500.csv"
df_origin = pd.read_csv(url)

# 1. Імпорт та первинне дослідження

COLUMNS_TO_DROP =[] # це константа, змінна яка змінюватись не буде

print("\n------ head ------")
print(df_origin.head())

print("\n------ info ------")
print(df_origin.info())

print("\n------ describe ------")
print(df_origin.describe())

print("\n------ describe for str ------")
print(df_origin.describe(include=[object]).T)

# Перевірити якість:

print('\n---- null -----')
# print(df.isna().sum()) # пропущені
print(df_origin.isna().sum().sort_values(ascending=False).head(20))

print('\n---- duplicated -----')
print(df_origin.duplicated().sum()) # дублікати шукаємо по рядках, чи не співпадають у нас повністю якісь рядки

print('\n---- List columns ------')
# list_col = df.columns
# print(list(list_col))
for i, col in enumerate(df_origin.columns):
    print(f"{i:02d}. {col}")

# 2. Очищення даних

# Видалити непотрібні колонки (за рішенням аналітика).

df = df_origin.copy()

if COLUMNS_TO_DROP:
    print("\n------- delete colums in list -----")
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')
else:
    print("\nCOLUMNS_TO_DROP = []")

# Стандартизувати формат текстових полів.

def standardize_text(s):
    if pd.isna(s):
        return np.nan
    
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()
    s = " ".join(s.split())

    return s


possible_email_cols = [c for c in df.columns if "email" in c.lower()]
possible_web_cols = [c for c in df.columns if ("web" in c.lower() or "website" in c.lower() or "url" in c.lower())]
possible_phone_cols = [c for c in df.columns if ("phone" in c.lower() or "telephone" in c.lower() or "tel" in c.lower())]
possible_fax_cols = [c for c in df.columns if "fax" in c.lower()]

print("\nPossible columns: ")
print("Email columns: ", possible_email_cols)
print("Web columns: ", possible_web_cols)
print("Phone columns: ", possible_phone_cols)
print("Fax columns: ",possible_fax_cols)

# Приміняємо зміни

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(standardize_text)

# Привести email та web до нижнього регістру.

# email
for col in possible_email_cols:
    df[col] = df[col].str.lower()

# web
for col in possible_web_cols:
    df[col] = df[col].str.lower()

# Очистити phone та fax від пробілів/символів.

def clean_phone(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.strip()
   
    plus = "+" if s.startswith("+") else ""

    digits = "" 
    for ch in s:
        if ch.isdigit():
            digits += ch

    digits = "".join(ch for ch in s if ch.isdigit())

    if digits == "":
        return np.nan
    
    return plus + digits

for col in possible_phone_cols + possible_fax_cols:
    df[col] = df[col].apply(clean_phone)


def title_if_str(s):
      if pd.isna(s):
        return np.nan
      return str(s).title()

city_cols = [c for c in df.columns if c.lower() in ("city", "city_name", "town")]

adress_cols = [c for c in df.columns if c.lower() in ("address")]

name_cols = [c for c in df.columns if c.lower() in ("name", "first_name", "second_name", "last_name", "company_name" )]

name_title = city_cols + adress_cols + name_cols

if name_title:
    for col in name_title:
        df[col] = df[col].apply(title_if_str)
    print("\n------ name of title -------")    
else:
    print("\n------ haven't name -------")  

print(df.head())

# 3. Створення нових колонок (Feature Engineering)
# Нова колонка	Опис
# full_name	Ім’я + прізвище
# email_domain	Домен email
# city_length	Довжина назви міста
# is_gmail	Boolean: чи email з gmail.com
