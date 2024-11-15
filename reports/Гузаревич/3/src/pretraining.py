import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

maternal_health_risk = fetch_ucirepo(id=863)

X = maternal_health_risk.data.features

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(maternal_health_risk.data.targets.values.ravel())
y = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class AutoencoderPretrainer:
    def __init__(self, input_dim, encoding_dims, epochs=50, batch_size=16):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder_weights = []
        self.encoder_bias = []

    def pretrain_layer(self, X, layer_index):
        input_layer = Input(shape=(X.shape[1],))
        encoded = Dense(self.encoding_dims[layer_index], activation='relu')(input_layer)
        decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(), loss='mse')

        autoencoder.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        encoder = Model(inputs=input_layer, outputs=encoded)
        self.encoder_weights.append(encoder.get_weights())

        X_encoded = encoder.predict(X)

        return X_encoded

    def pretrain(self, X):
        for i in range(len(self.encoding_dims)):
            print(f"\nПредобучение слоя {i + 1} с размерностью {self.encoding_dims[i]}")
            X = self.pretrain_layer(X, i)

    def build_final_model(self, output_dim):
        inputs = Input(shape=(self.input_dim,))
        x = inputs

        for i, encoding_dim in enumerate(self.encoding_dims):
            layer = Dense(encoding_dim, activation='relu')
            x = layer(x)
            layer.set_weights(self.encoder_weights[i])

        outputs = Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

pretrainer = AutoencoderPretrainer(input_dim=X_train.shape[1], encoding_dims=[128, 64, 32], epochs=100, batch_size=32)
pretrainer.pretrain(X_train)

final_model = pretrainer.build_final_model(output_dim=3)
final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training = final_model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=1)

y_pred_prob = final_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

plt.plot(training.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Pre-training')
plt.legend()
plt.show()