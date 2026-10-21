# Sprawozdanie - Laboratorium 1

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 1 / Sekwencje i sygnał mowy: od fali dźwiękowej do reprezentacji

**Autor / grupa:** Maciej Próchniak / gr. 3

**Data:** 21.10.2026

## 1. Cel ćwiczenia

Celem ćwiczenia było zamienienie trzech własnych nagrań mowy na reprezentacje log-mel i MFCC, porównanie ich między nagraniami oraz sprawdzenie, jak długość okna (n_fft) i krok ramkowania (hop_length) zmieniają spektrogram. Nagrania mają rosnącą złożoność: samogłoska „a”, wyraz „muzyka” i zdanie „Mały pies głośno szczekał, kiedy zobaczył kota”.

## 2. Metoda i implementacja

Nagrania mp3 wczytano przez librosa.load do 16 kHz mono. Log-mel liczono przez melspectrogram i power_to_db(ref=np.max), a MFCC z tej samej macierzy (mfcc(S=S_db)), więc obie reprezentacje mają wspólną oś czasu. Ustawienia podstawowe: n_fft = 512 (32 ms), hop_length = 160 (10 ms), 64 pasma mel, 13 współczynników MFCC. W eksperymentach zmieniano jeden parametr naraz: n_fft = 256, 512, 1024 oraz hop_length = 80, 160, 320, przy wspólnej skali barw od -80 do 0 dB.

**Kod.** Cały kod znajduje się w pliku `lab1_Maciej_Próchniak.ipynb` w repozytorium: https://github.com/mprochniak1/Speech_Recognition

## 3. Wyniki

| Nagranie | Czas [s] | Mowa [s] | Log-mel | MFCC |
|---|---|---|---|---|
| samogłoska „a” | 1,82 | 0,16-0,44 | [64, 183] | [13, 183] |
| wyraz „muzyka” | 1,82 | 0,17-0,86 | [64, 183] | [13, 183] |
| zdanie | 3,61 | 0,16-2,77 | [64, 362] | [13, 362] |

**Średnie MFCC (c0-c4) w wybranych fragmentach**

| Fragment | Czas [s] | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|---|
| „a” - samogłoska | 0,20-0,30 | -230,3 | 103,6 | -2,2 | -23,1 | -25,8 |
| „a” - muzyka | 0,64-0,76 | -304,1 | 111,0 | -3,4 | -2,5 | -2,2 |
| „a” - mały | 0,26-0,32 | -248,1 | 88,8 | -24,4 | -23,5 | -35,8 |
| „a” - zobaczył | 2,08-2,13 | -283,0 | 123,6 | -3,4 | -11,6 | -14,5 |
| „u” - muzyka | 0,24-0,34 | -346,5 | 98,3 | 34,1 | 5,6 | -12,9 |
| „y” - muzyka | 0,46-0,51 | -285,4 | 66,2 | 12,2 | 21,2 | 20,0 |
| „ś” - głośno | 0,85-0,89 | -335,1 | -55,3 | 7,4 | 33,6 | 20,5 |
| „sz” - szczekał | 1,00-1,07 | -341,7 | -60,2 | 9,4 | 22,9 | 15,3 |
| „z” - muzyka | 0,38-0,44 | -315,5 | -25,9 | 63,3 | -1,1 | 13,6 |
| cisza | 1,00-1,50 | -640,0 | 0,0 | 0,0 | 0,0 | 0,0 |

Przy n_fft = 256, 512 i 1024 STFT zdania ma 129, 257 i 513 prążków, a log-mel zawsze [64, 362] (Rys. 4). Przy hop_length = 80, 160 i 320 liczba ramek wynosi 365, 183 i 92 dla nagrań 1,82 s oraz 723, 362 i 181 dla zdania (Rys. 5), zgodnie ze wzorem T = 1 + ⌊N / hop_length⌋.

## 4. Analiza i interpretacja

**Samogłoska „a” (Rys. 1).** Obraz jest prawie stały w czasie: przez cały fragment 0,17-0,42 s widać poziome prążki harmonicznych i jasne pasmo ok. 500-1200 Hz (formanty „a”), a kolumny MFCC są prawie jednakowe.

**Wyraz „muzyka” (Rys. 2).** W „mu” (ok. 0,18-0,36 s) prążki tworzą łuk, czyli ton najpierw rośnie, a potem spada. Dźwięczne „z” (ok. 0,37-0,45 s) ma szum powyżej ok. 3 kHz i jednocześnie prążki na dole. Przed końcowym „a” widać ciemny pas zwarcia „k” (ok. 0,53-0,57 s) i krótki wybuch.

**Zdanie (Rys. 3).** Widać pauzę po przecinku (ok. 1,51-1,76 s), jasne plamy szumu w górnych pasmach przy „s”, „ś”, „sz” i „cz” (np. „szczekał” ok. 1,00-1,13 s) oraz krótkie ciemne zwarcia przed „p”, „k” i „t” (np. „t” w „kota” ok. 2,48-2,53 s). Ton jest wysoki w „mały” (rzadkie prążki ok. 0,34-0,44 s) i opada w „szczekał” przed przecinkiem. Na MFCC w zdaniu wyraźnie widać tylko głoski szumowe (niebieski c1) i pauzę, a samogłoski trudno odróżnić wzrokowo.

**MFCC (tabela).** Wszystkie „a” mają duży dodatni c1 (88,8-123,6) i ujemne c2-c4, mimo różnego tonu. „y” ma mniejszy c1 i dodatnie c3 i c4, a bezdźwięczne „ś” i „sz” ujemny c1 (ok. -55 i -60), bo ich energia leży w wysokich częstotliwościach. Dźwięczne „z” ma c1 tylko -25,9, bo ma też energię w niskich pasmach. W ciszy c0 = -640, a reszta współczynników jest równa 0.

**Wpływ n_fft (Rys. 4).** Okno 16 ms dobrze pokazuje granice w czasie (szeroki, ciemny pas zwarcia „k” w „muzyka”), ale harmoniczne „a” zlewają się w plamę. Okno 64 ms rozdziela prążki (np. łuk w „mu”), ale rozmywa granice: „a” zaczyna się na wykresie ok. 0,15 s zamiast 0,17 s, a zwarcia się zwężają. Okno 32 ms wydaje się rozsądnym kompromisem.

**Wpływ hop_length (Rys. 5).** Krok zmienia tylko gęstość ramek, a nie ostrość prążków. Przy 20 ms obraz robi się schodkowy, a przy 5 ms gładszy, ale bez nowych szczegółów.

**Ograniczenia.** Granice głosek odczytano z wykresów (ok. 10-20 ms dokładności), a nagrania pochodzą od jednej osoby i mają prawie cyfrową ciszę.

## 5. Wnioski

Log-mel jest czytelniejszy wzrokowo niż MFCC: widać na nim harmoniczne, formanty, szumy, zwarcia i pauzy. MFCC (13 zamiast 64 wartości na ramkę) dobrze oddzielają głównie głoski dźwięczne od szumowych i ciszę, a różnice między samogłoskami widać raczej w liczbach niż na wykresie. W pojedynczej samogłosce widać głównie barwę głoski, a w wyrazie i zdaniu dochodzą zmiany w czasie: zwarcia, przejścia między głoskami i intonacja, więc kolejność ramek niesie istotną informację. n_fft wyznacza kompromis między rozdzielczością w czasie i w częstotliwości, a hop_length zmienia głównie liczbę ramek.
