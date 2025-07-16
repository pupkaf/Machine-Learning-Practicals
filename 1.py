import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Создание серий
A = pd.Series({ 
    "0": "a", 
    "1": "a", 
    "2": "b", 
    "3": "a" 
}) 
B = pd.Series({ 
    "0": "b", 
    "1": "b", 
    "2": "a", 
    "3": "b" 
}) 
C = pd.Series({ 
    "0": "0", 
    "1": "0", 
    "2": "1", 
    "3": "0" 
}) 

# Создание DataFrame
df = pd.DataFrame({ 
    "A": A, 
    "B": B, 
    "Class": C 
}) 
try:
    df_csv = pd.read_csv('DataSetRadAndDP10000 PC1.csv') 
    print("\nДанные из CSV-файла:")
    print(df_csv.head())
except FileNotFoundError:
    print("\nФайл 'DataSetRadAndDP10000 PC1.csv' не найден.")

# Разделение DataFrame на две части
table_one = df_csv.iloc[:, :5]  # Первые 2 столбца
table_two = df_csv.iloc[:, 5:]   # Остальные столбцы
print("\nПервая таблица:")
print(table_one.head())
print("\nВторая таблица:")
print(table_two.head())
merged_table = pd.merge(table_one, table_two, left_index=True, right_index=True)
print("\nОбъединенная таблица:")
print(merged_table.head())
merged_table = pd.merge(table_one, table_two, left_index=True, right_index=True)
matrix = np.array(merged_table)
print("\nДвумерная матрица:")
print(matrix)
plt.figure(figsize=(10, 6))
plt.plot(merged_table.iloc[:, 1], merged_table.iloc[:, 2], marker='o', linestyle='-', color='b', label='Зависимость')
plt.title('График зависимости второго столбца от третьего')
plt.xlabel('Третий столбец')
plt.ylabel('Второй столбец')
plt.xlim(0, 1)  
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()


