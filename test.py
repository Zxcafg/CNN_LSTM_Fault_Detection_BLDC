# /**
#  * @file test.py
#  * @author Pavel Tshonek
#  * @date 2025
#  * @brief Skrypt do predykcji stanu silnika na podstawie wytrenowanego modelu.
#  *
#  * Skrypt ładuje wytrenowany model keras (CNN-LSTM) oraz przetwarza pliki CSV z pomiarami.
#  * Dane są normalizowane Z-score z użyciem skalery zapisanych z treningu, a następnie dzielone na sekwencje.
#  * Model wykorzystuje 5 cech wejściowych: I_Q, I_D, V_Q, V_D, is_startup.
#  * is_startup = 0 dla stanu ustalonego, is_startup = 1 dla rozruchu.
#  * Model zwraca prawdopodobieństwo uszkodzenia, na podstawie którego klasyfikowany jest stan silnika.
#  */

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

class MotorFaultPredictor:
    def __init__(self):
        # Ładowanie modelu (oczekuje 5 cech wejściowych)
        self.model = tf.keras.models.load_model('best_motor_model.keras')
        # Ładowanie zapisanych skalery z pliku treningowego
        self.scalers = joblib.load('scalers.pkl')

    def prepare_sequence_data(self, data, sequence_length):
        """
        Przygotowuje dane sekwencyjne za pomocą okna przesuwnego (sliding window).
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
        Przetwarza dane wejściowe: normalizacja Z-score (używając skalery z treningu) i tworzenie sekwencji.
        Args:
            data_df: DataFrame z kolumnami I_Q, I_D, V_Q, V_D, is_startup
            sequence_length: długość sekwencji dla modelu
        Returns:
            tablica numpy z sekwencjami (5 cech) gotowymi do podania na wejście modelu
        """
        features = ['I_Q', 'I_D', 'V_Q', 'V_D']
        scaled_data = np.zeros((len(data_df), 5), dtype=float)

        # Normalizacja Z-score każdej cechy ciągłej - używamy transform(), a nie fit_transform()!
        for idx, feature in enumerate(features):
            scaled_data[:, idx] = self.scalers[feature].transform(
                data_df[feature].values.reshape(-1, 1)
            ).ravel()

        # Kolumna 'is_startup' (0 - stan ustalony, 1 - rozruch) - bez normalizacji
        scaled_data[:, 4] = data_df['is_startup'].values
        sequences = self.prepare_sequence_data(scaled_data, sequence_length)

        return sequences

    def predict(self, data_path):
        """
        Główna funkcja predykcji. Wczytuje plik, przetwarza dane i dokonuje klasyfikacji.
        Args:
            data_path: ścieżka do pliku CSV z pomiarami
        Returns:
            result: string "DAMAGED MOTOR" lub "NORMAL MOTOR"
            confidence: poziom ufności predykcji w procentach
        """
        try:
            data = pd.read_csv(data_path)
            processed_data = self.preprocess_data(data)
            predictions = self.model.predict(processed_data)

            # Obliczenie średniej predykcji dla wszystkich sekwencji z pliku
            final_prediction = np.mean(predictions)
            if final_prediction >= 0.5:
                result = "DAMAGED MOTOR"
                confidence = final_prediction * 100
            else:
                result = "NORMAL MOTOR"
                confidence = (1 - final_prediction) * 100

            print(f"\nAnalysis Results for {data_path}:")
            print(f"Classification: {result}")
            print(f"Confidence: {confidence:.2f}%")

            return result, confidence

        except Exception as e:
            print(f"Error processing file {data_path}: {str(e)}")
            return None, None

if __name__ == "__main__":
    # Create predictor instance
    predictor = MotorFaultPredictor()

    # Definicja ścieżek do danych testowych (zbiór testy2)
    base_paths = ['stannormalny10k', 'stannormalny15k','uszkodzenia9k']
    file_types = ['ustalony', 'rozruch']
    test_files = [
        f'test/{base}/{type}{i}.csv'
        for base in base_paths
        for type in file_types
        for i in range(1, 6)
    ]

    for file in test_files:
        predictor.predict(file)