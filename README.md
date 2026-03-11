# Detekcja uszkodzeń silnika BLDC z wykorzystaniem głębokiego uczenia

## Opis projektu
Projekt realizuje detekcję uszkodzeń silnika bezszczotkowego prądu stałego (BLDC) na podstawie pomiarów prądów i napięć w osiach **Q** i **D** (przestrzeń wektorowa po transformacjach Clarke'a i Parka). Pomiary zostały pobrane z trzech faz silnika sterowanego za pomocą **FOC (Field Oriented Control)** przez układ ESC (Electronic Speed Controller) zbudowany na bazie **mikrokontrolera STM32** (rodzina STM32F4). Zastosowanie transformacji Clarke'a i Parka pozwoliło na przekształcenie prądów i napięć z trójfazowego układu współrzędnych (ABC) do wirującego układu współrzędnych (DQ), co upraszcza analizę i sterowanie silnikiem.

Wykorzystano hybrydowy model **CNN-LSTM**, który analizuje sekwencje czasowe pomiarów i klasyfikuje stan silnika jako **zdrowy (normal)** lub **uszkodzony (damaged)**.

Pomiary zostały zebrane dla różnych prędkości obrotowych (5k, 6k, 7k, 9k, 10k, 15k RPM) w dwóch stanach pracy: **rozruch** (pierwsze 15000 próbek) oraz **stan ustalony** (próbki 19500-25300). Dane te posłużyły do trenowania, walidacji i testowania modelu.

## Autor
- **Pavel Tshonek**

## Architektura modelu
Model zbudowany jest w oparciu o framework TensorFlow/Keras i składa się z następujących warstw:

1.  **Conv1D (128 filtrów, kernel=3)** - wyodrębnianie lokalnych cech z sekwencji.
2.  **BatchNormalization + MaxPooling1D** - normalizacja i redukcja wymiarowości.
3.  **Dropout (20%)** - regularyzacja, zapobieganie przeuczeniu.
4.  **LSTM (128 jednostek)** - uczenie zależności długoterminowych w sekwencjach.
5.  **BatchNormalization + Dropout (30%)**
6.  **Dense (64 neurony, aktywacja ReLU)** - warstwa w pełni połączona.
7.  **BatchNormalization + Dropout (40%)**
8.  **Dense (1 neuron, aktywacja sigmoid)** - klasyfikacja binarna.

<img width="1574" height="915" alt="image" src="https://github.com/user-attachments/assets/a8d0bfe8-d7ae-419f-91e0-b3e57219b744" />

## Struktura repozytorium
```
.
├── 1CNN_1LSTM_1DENSE.py      # Skrypt do trenowania modelu
├── test.py                    # Skrypt do predykcji na nowych danych
├── obcinanie.m                # (MATLAB) Przycinanie danych do rozruchu/stanu ustalonego
├── sprawdzenie.m              # (MATLAB) Wizualizacja surowych danych
├── pomiary/                    # Surowe dane pomiarowe z ESC
│   ├── stannormalny5k/
│   ├── stannormalny6k/
│   ├── stannormalny7k/
│   ├── uszkodzenia5k/
│   ├── uszkodzenia6k/
│   └── uszkodzenia7k/
├── testy_is_startup_raw/       # Dane treningowe z kolumną is_startup (po przetworzeniu)
│   ├── stannormalny5k/
│   ├── stannormalny6k/
│   ├── stannormalny7k/
│   ├── uszkodzenia5k/
│   ├── uszkodzenia6k/
│   └── uszkodzenia7k/
└── test/                       # Dane testowe z kolumną is_startup (po przetworzeniu)
    ├── stannormalny10k/
    ├── stannormalny15k/
    └── uszkodzenia9k/
```

## Akwizycja danych i przetwarzanie wstępne

### Stanowisko pomiarowe
Stanowisko pomiarowe składa się z:
- **Silnik BLDC** - badany obiekt
- **Układ ESC** z mikrokontrolerem **STM32G4** - realizuje sterowanie FOC (Field Oriented Control) i akwizycję danych
- **Czujniki prądu i napięcia** - pomiary w trzech fazach silnika
- **Interfejs komunikacyjny** - przesyłanie danych do komputera

<img width="1616" height="825" alt="image" src="https://github.com/user-attachments/assets/9e9c45ee-21a3-4b58-b3fb-a17d34a8d8ec" />


### Pomiary z silnika BLDC
Pomiary zostały pobrane z trzech faz silnika BLDC sterowanego za pomocą **FOC (Field Oriented Control)** przez układ ESC (Electronic Speed Controller) zbudowany na bazie **mikrokontrolera STM32** (rodzina STM32F4). W układzie tym w czasie rzeczywistym wykonywane są:

1. **Pomiar prądów fazowych** - poprzez rezystory bocznikowe
2. **Pomiar napięć fazowych** - poprzez dzielniki napięcia
3. **Transformacja Clarke'a** - przekształcenie prądów i napięć z układu trójfazowego (ABC) do układu stacjonarnego (αβ)
4. **Transformacja Parka** - przekształcenie wartości z układu stacjonarnego (αβ) do wirującego układu współrzędnych (DQ)

Wynikiem tych operacji są cztery wielkości w przestrzeni wektorowej Q-D:
- **I_Q** - prąd w osi Q (momentotwórczy)
- **I_D** - prąd w osi D (strumieniotwórczy)
- **V_Q** - napięcie w osi Q
- **V_D** - napięcie w osi D

Wszystkie te wielkości są zapisywane bezpośrednio do plików CSV z **częstotliwością próbkowania 1 kHz**. Struktura zapisywanych danych wygląda następująco:

```
TIMESTAMPS, I_Q_MEAS, I_D_MEAS, V_Q, V_D
1020421, 0.0202004, 0.113857, -0.00539168, -0.0293547
1020422, -0.00734559, -0.00367279, 0, -0.00898614
1020423, -0.00734559, 0.0624375, 0.000599076, -0.0257603
1020424, -0.0661103, 0.0293824, 0.016175, -0.0224653
```

### Przygotowanie danych treningowych
Surowe pliki CSV z ESC zawierają długie rejestracje pomiarów. Do trenowania modelu potrzebne były próbki o równej długości, dlatego skrypt `obcinanie.m`:

- Wycina pierwsze **15000 próbek** (15 sekund przy częstotliwości 1 kHz) jako reprezentację **stanu rozruchu**
- Wycina próbki od **19500 do 25300** (około 5.8 sekundy) jako reprezentację **stanu ustalonego**

Dodatkowo skrypt dodaje kolumnę **is_startup**:
- **is_startup = 1** dla danych rozruchowych
- **is_startup = 0** dla danych stanu ustalonego

## Zbiory danych

### Dane treningowe
- **Silnik zdrowy (label 0):** 30 plików (5 plików × 2 typy × 3 prędkości: 5k, 6k, 7k)
- **Silnik uszkodzony (label 1):** 30 plików (5 plików × 2 typy × 3 prędkości: 5k, 6k, 7k)
- **Typy plików:** `ustalonyX.csv` (stan ustalony, is_startup=0), `rozruchX.csv` (stan rozruchu, is_startup=1)

### Dane testowe
- **Silnik zdrowy:** 20 plików (5 plików × 2 typy × 2 prędkości: 10k, 15k)
- **Silnik uszkodzony:** 10 plików (5 plików × 2 typy × 1 prędkość: 9k)

## Przetwarzanie danych (Preprocessing)

### Analiza wstępna i ekstrakcja fragmentów
Przed przystąpieniem do przycinania danych, wykonano wstępną analizę wizualną surowych pomiarów za pomocą skryptu `sprawdzenie.m`, który pozwolił na ocenę charakteru sygnałów i identyfikację fragmentów reprezentujących stan rozruchu oraz stan ustalony.

Na podstawie tej analizy określono zakresy próbek do ekstrakcji:
- **Stan rozruchu** - pierwsze 15000 próbek (od początku rejestracji)
- **Stan ustalony** - próbki od 19500 do 25300 (około 5800 próbek)

Wybór tych konkretnych zakresów zapewnił, że wszystkie wycięte fragmenty mają jednakową długość w ramach każdej kategorii, co jest kluczowe dla procesu trenowania modeli głębokiego uczenia.

Skrypt `obcinanie.m` realizuje przycinanie surowych plików CSV oraz dodaje kolumnę **is_startup**:
- **is_startup = 1** dla danych rozruchowych
- **is_startup = 0** dla danych stanu ustalonego

### Normalizacja danych
Każda z czterech cech ciągłych (`I_Q`, `I_D`, `V_Q`, `V_D`) jest normalizowana osobno za pomocą **normalizacji Z-score** z wykorzystaniem `StandardScaler` z biblioteki scikit-learn:

```python
# Normalizacja Z-score każdej cechy (I_Q, I_D, V_Q, V_D)
for index in range(len(features)):
    current_feature = features[index]
    scaled_data[:, index] = self.scalers[current_feature].fit_transform(
        data_df[current_feature].values.reshape(-1, 1)
    ).ravel()
```

Dla każdej cechy obliczana jest średnia i odchylenie standardowe na zbiorze treningowym, a następnie dane są przekształcane według wzoru:

$$z = \frac{x - \mu}{\sigma}$$

gdzie:
- **x** - wartość oryginalna
- **μ** - średnia cechy na zbiorze treningowym
- **σ** - odchylenie standardowe cechy na zbiorze treningowym

Wartości średniej i odchylenia standardowego są zapisywane do pliku `scalers.pkl` za pomocą `joblib`, co pozwala na zastosowanie tej samej normalizacji podczas predykcji na nowych danych (używając `transform()` zamiast `fit_transform()`).

**Uwaga:** Kolumna `is_startup` nie jest normalizowana, ponieważ zawiera wartości binarne (0 lub 1).

<img width="1433" height="904" alt="image" src="https://github.com/user-attachments/assets/fd633608-e482-4a22-b097-ee73f104b3d4" />

### Segmentacja na sekwencje
Po normalizacji dane są dzielone na sekwencje z użyciem okna przesuwnego (sliding window):
- **Długość sekwencji:** 100 kroków czasowych (100 ms przy częstotliwości próbkowania 1 kHz)
- **Przesunięcie okna:** 1 krok czasowy (maksymalne nakładanie się sekwencji)

Każda sekwencja ma kształt **(100, 5)** i stanowi jeden przykład treningowy dla modelu. Dzięki takiemu podejściu model CNN-LSTM może uczyć się zależności czasowych w sygnale, co jest kluczowe dla wykrywania wzorców charakterystycznych dla uszkodzeń silnika.
## Uruchomienie

### Trenowanie modelu
```bash
python 1CNN_1LSTM_1DENSE.py
```
Skrypt wczyta dane z katalogu `testy_is_startup_raw`, przetworzy je, zbuduje i wytrenuje model. Najlepszy model (pod względem `val_accuracy`) zostanie zapisany jako `best_motor_model.keras`, a skalery jako `scalers.pkl`. Na ekranie wyświetlone zostaną wykresy procesu uczenia oraz metryki na zbiorze testowym.

### Predykcja na nowych danych
```bash
python test.py
```
Skrypt załaduje wytrenowany model (`best_motor_model.keras`) oraz skalery (`scalers.pkl`), przetworzy wszystkie pliki zdefiniowane w zmiennej `test_files` i dla każdego z nich wypisze klasyfikację (DAMAGED/NORMAL) oraz poziom ufności.

## Wyniki
Model osiągnął zadowalającą dokładność na zbiorze testowym dla danych w stanie ustalonym oraz był zdolny do wykrywania uszkodzeń w stanie rozruchu silnika, potwierdzając skuteczność architektury CNN-LSTM w wykrywaniu uszkodzeń na podstawie sygnałów prądowo-napięciowych w przestrzeni wektorowej.

<img width="1464" height="726" alt="image" src="https://github.com/user-attachments/assets/709ca806-b7d8-4181-b58b-656df94dc2a1" />

<img width="1625" height="858" alt="image" src="https://github.com/user-attachments/assets/cfaf1587-f847-416c-9f6d-d9a1b8d14834" />
