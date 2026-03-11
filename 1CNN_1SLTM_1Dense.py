# /**
#  * @file 1CNN_1LSTM_1DENSE.py
#  * @author Pavel Tshonek
#  * @date 2025
#  * @brief Skrypt do trenowania hybrydowego modelu CNN-LSTM do detekcji uszkodzeń silnika BLDC.
#  *
#  * Skrypt ładuje dane z plików CSV (osobno dla stanu zdrowego i uszkodzonego),
#  * normalizuje je, dzieli na sekwencje, a następnie trenuje model.
#  * Architektura modelu: Conv1D -> MaxPooling -> LSTM -> Dense.
#  * Zapisuje najlepszy model i generuje wykresy procesu uczenia.
#  *
#  * UWAGA: W plikach CSV znajduje się kolumna 'is_startup' (0 - stan ustalony, 1 - rozruch),
#  * która jest używana jako dodatkowa cecha wejściowa. Model wykorzystuje więc 5 cech:
#  * I_Q, I_D, V_Q, V_D, is_startup.
#  */

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib  # Do zapisu skalery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


class MotorFaultDetector:
    def __init__(self):
        self.model = None
        # Skalery dla każdej cechy, aby móc później zastosować tę samą normalizację na danych testowych
        # Uwaga: is_startup nie jest normalizowane, bo to wartość binarna (0 lub 1)
        self.scalers = {
            'I_Q': StandardScaler(),
            'I_D': StandardScaler(),
            'V_Q': StandardScaler(),
            'V_D': StandardScaler()
        }
        self.history = None

    def build_model(self, sequence_length):
        """
        Buduje hybrydowy model CNN-LSTM z 5 cechami wejściowymi.
        Args:
            sequence_length: długość sekwencji wejściowej (liczba kroków czasowych)
        Returns:
            model: skompilowany model keras
        """
        model = tf.keras.Sequential([
            # Warstwa konwolucyjna do wyodrębniania lokalnych wzorców
            # input_shape = (sequence_length, 5) bo używamy 5 cech: I_Q, I_D, V_Q, V_D, is_startup
            Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(sequence_length, 5)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),  # Redukcja wymiarowości
            Dropout(0.2),  # Regularyzacja

            # Warstwa LSTM do uczenia zależności czasowych
            LSTM(128),
            BatchNormalization(),
            Dropout(0.3),

            # Gęsta warstwa do klasyfikacji
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')  # Wyjście binarne (0 - zdrowy, 1 - uszkodzony)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        return model

    def prepare_sequence_data(self, data, sequence_length):
        """
        Tworzy sekwencje z danych za pomocą okna przesuwnego.
        Args:
            data: tablica numpy z danymi (liczba_próbek, liczba_cech)
            sequence_length: długość sekwencji (okna)
        Returns:
            tablica numpy z sekwencjami (liczba_sekwencji, sequence_length, liczba_cech)
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    def preprocess_data(self, data_df, sequence_length=100):
        """
        Normalizuje dane (Z-score) i dzieli na sekwencje.
        Uwzględnia wszystkie 5 cech: I_Q, I_D, V_Q, V_D, is_startup.

        Args:
            data_df: DataFrame z kolumnami I_Q, I_D, V_Q, V_D, is_startup
            sequence_length: długość sekwencji dla modelu
        Returns:
            tablica numpy z sekwencjami (5 cech: I_Q, I_D, V_Q, V_D, is_startup)
        """
        features = ['I_Q', 'I_D', 'V_Q', 'V_D']
        # Tworzymy tablicę na 5 cech (4 znormalizowane + is_startup)
        scaled_data = np.zeros((len(data_df), 5), dtype=float)

        # Normalizacja Z-score każdej cechy (I_Q, I_D, V_Q, V_D)
        for index in range(len(features)):
            current_feature = features[index]
            scaled_data[:, index] = self.scalers[current_feature].fit_transform(
                data_df[current_feature].values.reshape(-1, 1)
            ).ravel()

        # Dodajemy kolumnę is_startup bez normalizacji (wartości binarne 0/1)
        scaled_data[:, 4] = data_df['is_startup'].values

        # Tworzenie sekwencji - używamy wszystkich 5 cech
        sequences = self.prepare_sequence_data(scaled_data, sequence_length)
        return sequences

    def train(self, X_train, X_test, y_train, y_test, epochs=15, batch_size=32):
        """
        Trenuje model z wykorzystaniem callbacków.
        - ReduceLROnPlateau: zmniejsza współczynnik uczenia, gdy walidacyjna dokładność przestaje rosnąć.
        - ModelCheckpoint: zapisuje model o najlepszej walidacyjnej dokładności.
        """
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_motor_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def plot_results(self):
        """
        Rysuje wykresy dokładności i straty w funkcji epoki.
        """
        if self.history is None:
            print("No training history available")
            return

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Definicja ścieżek do danych treningowych
    base_paths_healthy = ['stannormalny5k', 'stannormalny6k', 'stannormalny7k']
    base_paths_damaged = ['uszkodzenia5k', 'uszkodzenia6k', 'uszkodzenia7k']
    file_types = ['ustalony', 'rozruch']

    # Lista plików dla silnika zdrowego (label 0)
    # Pliki CSV zawierają kolumny: I_Q, I_D, V_Q, V_D, is_startup
    # is_startup = 0 dla stanu ustalonego, is_startup = 1 dla rozruchu
    healthy_files = [
        f'testy_is_startup_raw/{base}/{type}{i}.csv'
        for base in base_paths_healthy
        for type in file_types
        for i in range(1, 6)
    ]

    # Lista plików dla silnika uszkodzonego (label 1)
    damaged_files = [
        f'testy_is_startup_raw/{base}/{type}{i}.csv'
        for base in base_paths_damaged
        for type in file_types
        for i in range(1, 6)
    ]

    print("Starting data loading and preprocessing...")
    detector = MotorFaultDetector()
    sequence_length = 100

    # Przetwarzanie plików zdrowego silnika
    healthy_sequences = []
    for file in healthy_files:
        try:
            data = pd.read_csv(file)
            # Funkcja preprocess_data używa wszystkich 5 kolumn: I_Q, I_D, V_Q, V_D, is_startup
            sequences = detector.preprocess_data(data, sequence_length)
            healthy_sequences.append(sequences)
            print(f"Successfully processed healthy file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    # Przetwarzanie plików uszkodzonego silnika
    damaged_sequences = []
    for file in damaged_files:
        try:
            data = pd.read_csv(file)
            sequences = detector.preprocess_data(data, sequence_length)
            damaged_sequences.append(sequences)
            print(f"Successfully processed damaged file: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    # Łączenie wszystkich sekwencji w jeden zbiór danych
    X_healthy = np.vstack(healthy_sequences)
    X_damaged = np.vstack(damaged_sequences)
    X = np.vstack([X_healthy, X_damaged])
    y = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_damaged))])

    print(f"\nData loaded successfully!")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Healthy sequences: {len(X_healthy)}, Damaged sequences: {len(X_damaged)}")
    print(f"Number of features: {X.shape[2]} (I_Q, I_D, V_Q, V_D, is_startup)")

    # Podział na zbiór treningowy i testowy (80/20) z zachowaniem proporcji klas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Konwersja na float32 dla kompatybilności z TensorFlow
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    print("\nData shapes and types:")
    print(f"X_train: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_test: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_train: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"y_test: {y_test.shape}, dtype: {y_test.dtype}")

    print("\nBuilding and training the model...")
    detector.build_model(sequence_length)
    detector.model.summary()

    print("\nStarting training...")
    detector.train(X_train, X_test, y_train, y_test, epochs=15, batch_size=32)

    print("\nGenerating training plots...")
    detector.plot_results()

    print("\nEvaluating model on test data...")
    test_results = detector.model.evaluate(X_test, y_test, verbose=1)

    print("\nTest results:")
    for metric_name, value in zip(detector.model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")

    # Zapisanie skalery do pliku - będą potrzebne podczas predykcji na nowych danych
    print("\nSaving scalers for future predictions...")
    joblib.dump(detector.scalers, 'scalers.pkl')
    print("Scalers saved to 'scalers.pkl'")