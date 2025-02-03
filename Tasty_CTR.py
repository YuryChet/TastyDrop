import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.core.nanops import nanmax

# Настраиваем pandas на отображение всех столбцов без сокращений
pd.set_option('display.max_columns', None)  # Показывать все столбцы
pd.set_option('display.max_rows', None)  # Показывать все строки (осторожно: большие файлы)
pd.set_option('display.width', 1000)  # Настройка ширины для полного вывода строк

# путь
df = pd.read_csv('C:/Users/GoodM/Downloads/tasty_global_stat.csv')


# хочется еще добавить CR из клика в регистрацию
# из регистрации в первый депозит
# добавить просмотры


# ############## ПРЕДОБРАБОТКА ДАННЫХ в столбцах с результатами


# print(df.head())

#                                                   Предобработка числовых данных
# Дальше просто привожу данные к одному виду
columns_to_clean = ['Просмотров ролика', 'CPM140', 'CTR кликаб 0,5', 'CPC50', 'Клик2Рег', 'Рег2Деп']  # Задаем писок столбцов


# str- работает со строкой в стобике
# strip удаляет пробелы в конце и начале строки
# replace удаляет пустое значение и заменяет его на нан
# str- обращаемся к строке
# replae заменяет , на .
# Обрабатываем каждый столбец из списка
for col in columns_to_clean:
    df[col] = (
        df[col]
        .astype(str)              # Преобразуем значения в строки, чтобы работать с текстом
        .str.replace('\xa0', '')  # Удаляем неразрывные пробелы
        .str.strip()              # Убираем пробелы
        .replace('', np.nan)      # Заменяем пустые строки на NaN
        .replace(r'[^\d.,-]', '', regex=True)  # Удаляем нецифровые символы
        .str.replace(',', '.')    # Заменяем запятые на точки
    )
    # Заполняем оставшиеся NaN и преобразуем в float
    df[col] = pd.to_numeric(df[col], errors='coerce')






#######################################
#######################################
#               Предобработка строчки компании


# Все строчки компании к одному виду чтобы не париться
# strip - убирает пробелмы с конца и начала
# Меняем пустое на nan
# Убираем _
# Все бувы заглавные
df['Кампания'] = df['Кампания'].str.strip().replace('', np.nan).str.replace('_', '').str.upper()


df['Кампания'] = (
    df['Кампания']
    .str.strip()
    .str.replace('\xa0', '')  # Удаляем неразрывные пробелы
    .str.replace(r'[^\w\d]', '', regex=True)  # Удаляем ненужные символы
)
#                                                     Дополнительная предобработка
# тут просто заменим что кто-то растянул строчку c fr24
df['Кампания'] = df['Кампания'].str.replace(r'FORTUNE', 'FORTUNE23', regex=True)
df['Кампания'] = df['Кампания'].str.replace(r'SBC', 'SB23', regex=True)
df['Кампания'] = df['Кампания'].str.replace(r'NEWSCASE23', 'NEWCASE23', regex=True)
df['Кампания'] = df['Кампания'].str.replace(r'BF25', 'BF24', regex=True)
df['Кампания'] = df['Кампания'].str.replace(r'3CF32', '3CF24', regex=True)
df['Кампания'] = df['Кампания'].replace('FORTUNE2323', 'FORTUNE23')



#                                                    Предобработка строчки типа
# просто заменим преролл на интеграцию
df['Тип'] = df['Тип'].str.strip().replace('PreRoll', 'Интеграции')



#######################################
#######################################
#     Группировка получившихся данных

# Считаем среднее значение для метрик
grouped_metrics = df.groupby(['Кампания', 'Тип'])[['CPM140', 'CTR кликаб 0,5', 'CPC50', 'Клик2Рег', 'Рег2Деп']].median()
# print(grouped_metrics)

# считаем ПРОСМОТРЫ
grouped_metrics_views = df.groupby(['Кампания', 'Тип'])[['Просмотров ролика']].sum()
# print(grouped_metrics_views)

#связываем вместе
grouped_metrics = grouped_metrics.merge(grouped_metrics_views, on=['Кампания', 'Тип'], how='left')
# print(grouped_metrics)


#                                 Делаем так, чтобы была дата начала и конца
# переименовываем колонку в исходном датафрейме
df.rename(columns={df.columns[0]: 'Дата'}, inplace=True)





# Функция для обработки строковых дат
# Это нужно было так как год где-то был как просто 24 а не 2024 по этому распаршивался далее неправильно
def fix_year(date_str):
    if isinstance(date_str, str) and len(date_str.split('.')) == 3:
        day, month, year = date_str.split('.')  # Разделяем день, месяц, год
        if len(year) == 2:  # Если год состоит из 2 цифр
            year = str(int(year) + 2000)  # Преобразуем в 4-значный год
        return f"{day}.{month}.{year}"  # Возвращаем обновлённую дату
    return date_str  # Если формат не строка, оставляем как есть



# Применяем функцию к столбцу
df['Дата'] = df['Дата'].apply(fix_year)
df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce', dayfirst=True)
df = df.sort_values("Дата")

# print(df)


# ФИЛЬТР ПО ДАТЕ
df = df[df['Дата'].dt.year > 2022]


#делаем группировку по компании и дате. Получаем минимальную и максимальную дату компании
grouped_dates = df.groupby(['Кампания', 'Тип'])['Дата'].agg(date_start='min', date_end='max').reset_index()
# print(grouped_dates)


# Объединение с исходным датафреймом, чтобы добавить столбцы date_start и date_end
df_groped = grouped_metrics.merge(grouped_dates, on=['Кампания', 'Тип'], how='left')









#                                 Работа с готовым датафреймом
#########################################

#переименовываем колонки внутри готового датафрейма

df_groped.rename(columns={'CPM140': 'CPM'}, inplace=True)
df_groped.rename(columns={'CTR кликаб 0,5': 'CTR'}, inplace=True)
df_groped.rename(columns={'CPC50': 'CPC'}, inplace=True)
df_groped.rename(columns={'Просмотров ролика': 'Просмотров'}, inplace=True)

df_groped = df_groped[(df_groped['CTR'] < 5) & (df_groped['CTR'].notna()) & (df_groped['CTR'] > 0.001)]
# print(df_groped)

# ОБРЕЗКА (округление) до 3 знаков
columns_to_trim = ['CTR']  # Укажите нужные столбцы

for col in columns_to_trim:
    df_groped[col] = df_groped[col].round(3)  # Обрезаем до 3 знаков после запятой


# ОБРЕЗКА (округление) до 2
columns_to_trim = ['CPM', 'CPC', 'Клик2Рег', 'Рег2Деп']  # Укажите нужные столбцы

for col in columns_to_trim:
    df_groped[col] = df_groped[col].round(2)  # Обрезаем до 3 знаков после запятой

# просмотры делаем целым числом
df_groped['Просмотров'] = df_groped['Просмотров'].astype(int)







# СОХРАНЕНИЕ ТЕКУЩЕГО ДАТАСЕТА
df_groped.to_csv('grouped_data.csv', index=True)







# СОРТИРОВКА и ТОПЫ
##################################
##################################

#### ТОПЫ ПО CTR

df_sorted_ctr = df_groped.sort_values(by='CTR', ascending=False)

#топ 10 по ctr интеграций
df_sorted_ctr_top_integration = (
    df_sorted_ctr
    .query('Тип == "Интеграции"')
    .iloc[:10]
)
# print(df_sorted_ctr_top_integration)


#топ 10 по ctr нарезки
df_sorted_ctr_top_narezki = (
    df_sorted_ctr
    .query('Тип == "Нарезки"')
    .iloc[:10]
)
# print(df_sorted_ctr_top_narezki)


#топ 10 по ctr отдельное видео
df_sorted_ctr_top_video = (
    df_sorted_ctr
    .query('Тип == "Отдельные видео"')
    .iloc[:10]
)

# print(df_sorted_ctr_top_video)



#                                ГРАФИКИ СТОЛБЧАТЫЕ
########################################

# ВКЛЮЧИТЬ
# столбчатый график
def plot_top_ctr(data, title):
    plt.figure(figsize=(12, 6))
    plt.bar(data['Кампания'], data['CTR'], color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel('Кампания', fontsize=12)
    plt.ylabel('CTR', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




#                                Запуск графиков
# График для интеграций  столбчатые
plot_top_ctr(df_sorted_ctr_top_integration, 'АНТИТОП-10 по CTR для Интеграций')

# График для нарезок
plot_top_ctr(df_sorted_ctr_top_narezki, 'АНТИТОП-10 по CTR для Нарезок')

# График для отдельных видео
plot_top_ctr(df_sorted_ctr_top_video, 'АНТИТОП-10 по CTR для Отдельных Видео')





#                                 ГРАФИКИ ЛИНЕЙНЫЕ
##################################################
# тут я хочу сделать график который бы показывал зависимость даты от резульата CTR

# сортировка графика по дате
df_sorted_date = df_groped.sort_values(by='date_start').reset_index(drop=True)

# Сортировка по дате и сброс индекса
df_sorted_date = df_groped.sort_values(by='date_start').reset_index(drop=True)
# print(df_sorted_date)
# Фильтруем данные для типа "Нарезки"
df_sorted_date_integration = df_sorted_date.query('Тип == "Нарезки"')
# print(df_sorted_date_integration)

# Создание меток для каждого месяца
months = pd.date_range(start=df_sorted_date_integration['date_start'].min(),
                       end=df_sorted_date_integration['date_start'].max(),
                       freq='MS')

# Построение графика
plt.figure(figsize=(16, 8))  # Увеличиваем размер графика
plt.plot(df_sorted_date_integration['date_start'], df_sorted_date_integration['CTR'], marker='o', linestyle='-', color='b')
plt.title('Зависимость CTR от даты НАРЕЗКИ', fontsize=16)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('CTR', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Устанавливаем метки для каждого месяца
plt.xticks(months, [date.strftime('%b %Y') for date in months], rotation=45, fontsize=10)

# Добавляем подписи к точкам (ближе к точкам)
for x, y, label in zip(df_sorted_date_integration['date_start'], df_sorted_date_integration['CTR'], df_sorted_date_integration['Кампания']):
    plt.text(x, y + 0.002, label, fontsize=8, color='black', ha='center', va='bottom', rotation=45)  # Подписываем около точек

plt.tight_layout()

# Показ графика
plt.show()









import matplotlib.dates as mdates

# Фильтруем данные по типу и сортируем их
df_narezki = df_groped.query('Тип == "Нарезки"').sort_values(by='date_start')
df_video = df_groped.query('Тип == "Отдельные видео"').sort_values(by='date_start')
df_integration = df_groped.query('Тип == "Интеграции"').sort_values(by='date_start')

# Проверяем наличие данных по просмотрам
# print("Проверка данных:")
# print("Нарезки:", df_narezki[['date_start', 'Просмотров']])
# print("Отдельные видео:", df_video[['date_start', 'Просмотров']])
# print("Интеграции:", df_integration[['date_start', 'Просмотров']])

# Построение графика
plt.figure(figsize=(16, 8))

# Линия для "Нарезки"
plt.plot(df_narezki['date_start'], df_narezki['Просмотров'], label='Нарезки', marker='o', linestyle='-', color='blue')

# Линия для "Отдельные видео"
plt.plot(df_video['date_start'], df_video['Просмотров'], label='Отдельные видео', marker='o', linestyle='-', color='green')

# Линия для "Интеграции"
plt.plot(df_integration['date_start'], df_integration['Просмотров'], label='Интеграции', marker='o', linestyle='-', color='orange')

# Настройки графика
plt.title('Зависимость просмотров от даты для разных типов', fontsize=16)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Просмотры', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Настройка отображения всех месяцев
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Метки для каждого месяца
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Формат меток: 'Месяц Год'

# Поворот подписей по оси X
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Показ графика
plt.show()
