import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import BernoulliRBM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('C:\\Users\\vadim\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Python 3.12\\Python Files\\Programs\\ИАД\\Raisin_Dataset.csv')  # Укажите правильный путь к вашему файлу

# Подготовка данных
X = data.drop('Class', axis=1)
y = data['Class']

# Преобразование целевой переменной в числовой формат
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Модель без предобучения
model1 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model1.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred1 = (model1.predict(X_test) > 0.5).astype("int32")
report1 = classification_report(y_test, y_pred1, output_dict=True)

print("Модель без предобучения:")
print(classification_report(y_test, y_pred1))

# 2. Модель с предобучением (автоэнкодер)
input_dim = X_train.shape[1]
encoding_dim = 16

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = tf.keras.Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

encoder_model = tf.keras.Model(inputs=input_layer, outputs=encoder)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]

model2 = Sequential([
    encoder_model,
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model2.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred2 = (model2.predict(X_test) > 0.5).astype("int32")
report2 = classification_report(y_test, y_pred2, output_dict=True)

print("Модель с предобучением (автоэнкодер):")
print(classification_report(y_test, y_pred2))

# 3. Модель с предобучением (RBM)
rbm = BernoulliRBM(n_components=16, learning_rate=0.01, n_iter=10, random_state=42)

X_train_rbm = rbm.fit_transform(X_train)
X_test_rbm = rbm.transform(X_test)

model3 = Sequential([
    Input(shape=(X_train_rbm.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model3.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history3 = model3.fit(X_train_rbm, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred3 = (model3.predict(X_test_rbm) > 0.5).astype("int32")
report3 = classification_report(y_test, y_pred3, output_dict=True)

print("Модель с предобучением (RBM):")
print(classification_report(y_test, y_pred3))

accuracy_without_pretraining = report1['accuracy']
accuracy_with_pretraining_autoencoder = report2['accuracy']
accuracy_with_pretraining_rbm = report3['accuracy']

print(f"Точность модели без предобучения: {accuracy_without_pretraining:.4f}")
print(f"Точность модели с предобучением (автоэнкодер): {accuracy_with_pretraining_autoencoder:.4f}")
print(f"Точность модели с предобучением (RBM): {accuracy_with_pretraining_rbm:.4f}")

# Построение графиков
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    
    ax1.plot(history.history['loss'], label='train loss')
    ax1.plot(history.history['val_loss'], label='val loss')
    ax1.set_title('Функция потерь')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Потери')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='train accuracy')
    ax2.plot(history.history['val_accuracy'], label='val accuracy')
    ax2.set_title('Точность')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Точность')
    ax2.legend()
    
    plt.show()

plot_history(history1, 'Модель без предобучения')
plot_history(history2, 'Модель с предобучением (автоэнкодер)')
plot_history(history3, 'Модель с предобучением (RBM)')
