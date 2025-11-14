from keras.datasets import cifar10
from keras import models, layers
import numpy as np

# Загрузка CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация
x_train, x_test = x_train / 255.0, x_test / 255.0

# Взять случайную подвыборку
indices = np.random.choice(len(x_train), 10000, replace=False)
x_train_small, y_train_small = x_train[indices], y_train[indices]

print(f"Полная выборка: {len(x_train)} изображений")
print(f"Случайная выборка: {len(x_train_small)} изображений")

# модель для сравнения
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 выходов для 10 классов
])

# Изменяем loss для целочисленных меток
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # для меток 0,1,2,...,9
    metrics=['accuracy']
)

# Тестируем на полной выборке (меньше эпох для скорости)
model.fit(
    x_train, y_train, 
    epochs=2, 
    batch_size=64, 
    validation_split=0.1, 
    verbose=1)

# Оценка полной точности
test_loss_full, test_acc_full = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность (полная): {test_acc_full:.4f}")

# Тестируем на нашей выборке
model.fit(x_train_small, y_train_small, epochs=2, batch_size=64, validation_split=0.1, verbose=1)

test_loss_small, test_acc_small = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность (случайная): {test_acc_small:.4f}")