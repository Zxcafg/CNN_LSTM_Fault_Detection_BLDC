% /**
%  * @file obcinanie.m
%  * @author Pavel Tshonek
%  * @date 2025
%  * @brief Skrypt do ekstrakcji i zapisu fragmentów stanu ustalonego i rozruchu z surowych plików CSV.
%  *
%  * Surowe pliki z ESC zawierają długie rejestracje. Do trenowania modeli ML potrzebne są
%  * próbki o równej długości. Skrypt wycina:
%  * - Stan rozruchu (startup_rows): pierwsze 15000 próbek.
%  * - Stan ustalony (steady_start:steady_end): próbki 19500 do 25300.
%  * Wynikowe dane zapisywane są jako osobne pliki CSV z nazwami rozruchX.csv i ustalonyX.csv.
%  * Dodawana jest również kolumna 'is_startup' (1 dla rozruchu, 0 dla stanu ustalonego).
%  */

clear all;
clc;

% Parametry przycinania danych
startup_rows = 15000;      % Liczba próbek dla stanu rozruchu (pierwsze 15000)
steady_start = 19500;      % Początek stanu ustalonego
steady_end = 25300;        % Koniec stanu ustalonego (daje około 5800 próbek)

% Przetwarzaj pliki od 1.csv do 5.csv
for file_num = 1:5
    % Nazwa pliku wejściowego
    filename = sprintf('%d.csv', file_num);
    
    % Wczytaj plik jako tabelę, pomijając pierwsze 11 linii (nagłówek ESC)
    opts = detectImportOptions(filename, 'NumHeaderLines', 11);
    opts.Delimiter = ',';
    T = readtable(filename, opts);
    
    % Konwersja tabeli na macierz numeryczną
    data = table2array(T);
    
    % Usuń pierwszą kolumnę (indeks czasu)
    data = data(:, 2:end);
    
    % Wyodrębnij dane dla stanu rozruchu - pierwsze N próbek
    startup_data = data(1:startup_rows, :);
    
    % Wyodrębnij dane dla stanu ustalonego - wycinek ze środka pliku
    steady_state_data = data(steady_start:steady_end, :);
    
    % DODANIE KOLUMNY is_startup
    % Dla danych rozruchowych dodaj kolumnę z wartościami 1
    startup_data = [startup_data, ones(size(startup_data, 1), 1)];
    % Dla danych stanu ustalonego dodaj kolumnę z wartościami 0
    steady_state_data = [steady_state_data, zeros(size(steady_state_data, 1), 1)];
    
    % Utwórz tabele z nazwami kolumn (teraz 5 kolumn)
    startup_table = array2table(startup_data, 'VariableNames', {'I_Q', 'I_D', 'V_Q', 'V_D', 'is_startup'});
    steady_state_table = array2table(steady_state_data, 'VariableNames', {'I_Q', 'I_D', 'V_Q', 'V_D', 'is_startup'});
    
    % Nazwy plików wyjściowych
    startup_filename = sprintf('rozruch%d.csv', file_num);
    steady_filename = sprintf('ustalony%d.csv', file_num);
    
    % Zapisz do plików CSV
    writetable(startup_table, startup_filename);
    writetable(steady_state_table, steady_filename);
    
    % Wyświetl postęp
    fprintf('Przetworzono plik %s:\n', filename);
    fprintf(' - Utworzono %s (dane rozruchowe: wiersze 1:%d, is_startup=1)\n', startup_filename, startup_rows);
    fprintf(' - Utworzono %s (dane stanu ustalonego: wiersze %d:%d, is_startup=0)\n\n', steady_filename, steady_start, steady_end);
end

disp('Przetwarzanie zakończone pomyślnie dla wszystkich plików 1.csv do 5.csv');