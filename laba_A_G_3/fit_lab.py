from keras.datasets import cifar10
from keras import models, layers
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test,y_test) = cifar10.load_data() # разделение данных на тренировку и тест
x_train,x_test = x_train/255.0, x_test/255.0
# Практически тоже самое
#x_train = x_train.reshape((60000, 28*28)).astype("float32")/255 # 255 - обычная нормализация для пикселей (6000 из-й)
#x_test = x_test.reshape((10000, 28*28)).astype("float32")/255

# One-Hot кодирование
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Взять случайную подвыборку (10000 изображений)
indices = np.random.choice(len(x_train), 10000, replace=False)
x_train_small, y_train_small = x_train[indices], y_train[indices]

# Вывод  
print(f"Полная выборка: {len(x_train)} изображений")
print(f"Случайная выборка: {len(x_train_small)} изображений")

model = models.Sequential([# последовательные слои модели (исп-е Sequential)
    layers.Conv2D(32,(3,3),activation='relu', input_shape = (32,32,3)),
    #input_shape-размер входного батча), relu - обеспечение нелинейности (для упрощения обучения)
    
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(), # Flatten распремляет
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam', # adam - умная версия градиентного спуска
    loss = 'categorial_crossentropy', #loss - функция потерь(есть много других вариантов, но эта просто самая легковесная)
)
# Полная выборка
history= model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size = 64, # batch_size = батчи, котрые поступают на вход(делим на 64)
    valitation_split = 0.1)


# Оценка на тестовой выборке
test_loss_full, test_acc_small = history.evaluate(x_test, y_test, verbose=0)
print(f"Точность на тесте (случайная выборка): {test_acc_small:.4f}")

# Сравнение результатов
print(f"Полная выборка: {test_loss_full:.4f}")
print(f"Случайная выборка (10000 изображений): {test_acc_small:.4f}")
print(f"Разница: {test_loss_full - test_acc_small:.4f}")