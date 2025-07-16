import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Загрузка датасета
data = pd.read_csv("Medical Text Dataset.csv", encoding='latin-1')

# Проверка данных
print(data.head())

# Переименование столбцов для удобства
data.columns = ['Category', 'Text']

# Преобразуем метки в числовой формат
data['Category'] = data['Category'].map({'Рак щитовидной железы': 0, 'Рак толстой кишки': 1, 'Рак легких': 2})

# Разделение данных на обучающую и тестовую выборки
X = data['Text']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование текста в числовые признаки с помощью CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели наивного байесовского классификатора
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test_vec)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Рак щитовидной железы', 'Рак толстой кишки', 'Рак легких'], yticklabels=['Рак щитовидной железы', 'Рак толстой кишки', 'Рак легких'])
plt.title('Confusion Matrix')
plt.xlabel('Предсказано')
plt.ylabel('Фактическое')
plt.show()

# Визуализация распределения категорий
plt.figure(figsize=(6, 4))
data['Category'].value_counts().plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Распределение категорий')
plt.xticks(ticks=[0, 1, 2], labels=['Рак щитовидной железы', 'Рак толстой кишки', 'Рак легких'], rotation=0)
plt.xlabel('Категория')
plt.ylabel('Количество')
plt.show()
