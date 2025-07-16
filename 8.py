import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Создание датасета
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=100)

# Разделение на обучающий и тестовый наборы
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Определение параметров для Random Search
param_distributions = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Создание модели RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Выполнение Random Search
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train, Y_train)

# Получение лучших параметров и лучшей оценки
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f'Лучшие параметры: {best_params}')
print(f'Лучшая точность: {best_score:.2f}')

# Оценка модели на тестовом наборе
Y_pred = random_search.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print(f'Точность на тестовом наборе: {test_accuracy:.2f}')

# Визуализация обучающего набора данных
plt.figure(figsize=(8, 8))
plt.scatter(X_train[Y_train == 0][:, 0], X_train[Y_train == 0][:, 1], color='blue', label='Класс 0', alpha=0.5)
plt.scatter(X_train[Y_train == 1][:, 0], X_train[Y_train == 1][:, 1], color='red', label='Класс 1', alpha=0.5)
plt.title('Обучающий набор данных')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()

# Визуализация тестового набора данных
plt.figure(figsize=(8, 8))
plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0][:, 1], color='blue', label='Класс 0', alpha=0.5)
plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1], color='red', label='Класс 1', alpha=0.5)
plt.title('Тестовый набор данных')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()
