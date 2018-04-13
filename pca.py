# PCA Principal Component Analysis
# Feature Extraction PCA сокращает количество зависимых переменных
# до 2, 3, ... которые несут наибольши вклад в результат предсказаний
# по факту он убираем каждый столбец и проверяет насколько сильно
# поменялись предсказания и потом каждый стортирует каждый столбец с
# наибольшего вклада в результат к наименьшему вкладу в предсказания
# предсказания реализуются линейной регрессией, которая с мин расстоянием
# до всех точек к каждому классу

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# убрали варнинг используя эту библиотеку model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # модифицируем и анализируем данные
X_test = sc_X.transform(X_test) # модифицируем только X_test без анализа

# Applying PCA
from sklearn.decomposition import PCA
# алгоритм такой, сначала ставим в параметр n_components = None, а потом
# смотрим минимальное количество столбцов с приемлемой точностью и ставим 2
pca = PCA(
  n_components = 2 # (2) количество переменных которых надо оставить
) # None будет отсортированный вектор по анализу вклада каждой из колонок
# считает какой вклад вносит каждая колонка в общий результат
# 0 = 36.9%, 1 = 19.8%... вклад, если взять 0 и 1, то будет 36.9+19.8=56.7%
# мы нашли что 2 нас устраивает и меняем None => 2
X_train = pca.fit_transform(X_train) # оставляем только 2 параметра + scale
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # инициализация модели
classifier.fit(X_train, y_train) # закидываем в модель данные для обучения модели

# Predicting the Test set results
y_pred = classifier.predict(X_test) # предсказываем данные из X_test

# Making the Confusion Matrix # узнаем насколько правильная модель
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # закиыдваем тестовые и предсказанные данные
# данные в 3х3 матрице [[14 0 0], [1, 15, 0], [0, 0, 6]]
# 1 тут говорит о том, что реальный пользователь был в 0 группе, а показал 1


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# подготавливаем матрицу для нашего поля данных с шагом сетки 0.01
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# вся магия тут, раскрашиваем данные по всему полотку X1, X2
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

# границы для областей указываем?
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# все точки рисуем на полотне, которые у нас есть
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Principal Component Analysis (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

# подготавливаем матрицу для нашего поля данных с шагом сетки 0.01
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# вся магия тут, раскрашиваем данные по всему полотку X1, X2
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

# границы для областей указываем?
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# все точки рисуем на полотне, которые у нас есть
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Principal Component Analysis (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()