% /**
%  * @file sprawdzenie.m
%  * @author Pavel Tshonek
%  * @date 2025
%  * @brief Skrypt do wizualizacji surowych danych pomiarowych z silnika.
%  *
%  * Skrypt wczytuje pojedynczy plik CSV, oblicza moduły prądu i napięcia
%  * (sqrt(I_Q^2 + I_D^2) oraz sqrt(V_Q^2 + V_D^2)) i rysuje je na wspólnym wykresie.
%  * Służy do szybkiego podglądu charakteru sygnałów.
%  */

clear all;
clc;

% Nazwa  przykładowego pliku do analizy
filename = 'pomiary/uszkodzenia7k/5.csv';

% Wczytaj plik jako tabelę, pomijając pierwsze 11 linii (nagłówek ESC)
% separator to przecinek
opts = detectImportOptions(filename, 'NumHeaderLines', 11);
opts.Delimiter = ',';
T = readtable(filename, opts);

% Konwersja tabeli na macierz numeryczną
data = table2array(T);

% Usuń pierwszą kolumnę (indeks czasu)
data = data(:, 2:end);

% Przypisz kolumny do zmiennych
I_Q = data(:, 1);  % Prąd w osi Q
I_D = data(:, 2);  % Prąd w osi D
V_Q = data(:, 3);  % Napięcie w osi Q
V_D = data(:, 4);  % Napięcie w osi D

% Oblicz moduły prądu i napięcia (długość wektora w przestrzeni Q-D)
current = sqrt(I_Q.^2 + I_D.^2);
voltage = sqrt(V_Q.^2 + V_D.^2);

% Wektor czasu (numery próbek)
time = (0:length(current)-1)';

% Rysuj wykres
figure;
yyaxis left
plot(time, current, 'b-', 'LineWidth', 1.5);
ylabel('Prąd (A)');

yyaxis right
plot(time, voltage, 'r--', 'LineWidth', 1.5);
ylabel('Napięcie (V)');

xlabel('Numer próbki');
title('Moduły prądu i napięcia');
legend('Prąd', 'Napięcie');
grid on;