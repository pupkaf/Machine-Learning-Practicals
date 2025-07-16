import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV

# Создание датасета
X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=100)

# Разделение на обучающий и тестовый наборы
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Определение параметров для Байесовской оптимизации
param_space = {
    'n_estimators': (50, 200),  # диапазон значений для n_estimators
    'max_depth': (3, 10),        # диапазон значений для max_depth
    'min_samples_split': (2, 10) # диапазон значений для min_samples_split
}

# Создание модели RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Выполнение Байесовской оптимизации
opt = BayesSearchCV(model, param_space, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1)
opt.fit(X_train, Y_train)

# Получение лучших параметров и лучшей оценки
best_params = opt.best_params_
best_score = opt.best_score_

print(f'Лучшие параметры: {best_params}')
print(f'Лучшая точность: {best_score:.2f}')

# Оценка модели на тестовом наборе
Y_pred = opt.predict(X_test)
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
