# Oxford-IIIT Pet Classification - Custom CNN

Projekt ten implementuje od zera konwolucyjną sieć neuronową (CNN) w PyTorch, której celem jest klasyfikacja 37 ras psów i kotów na podstawie zbioru danych Oxford-IIIT Pet Dataset.

Projekt miał na celu zbadanie wpływu różnych technik regularyzacji i augmentacji danych na proces uczenia głębokiego z użyciem małego zbioru danych i własnej architektury sieci.

## Architektura Modelu (`PetNet`)

Model został zbudowany od podstaw przy użyciu architektury blokowej. Przekształca on wejściowy obraz o wymiarach `3x128x128` w wektor prawdopodobieństw dla 37 klas.

* **Blok 1 & 2:** Podwójne warstwy konwolucyjne (`Conv2d`) oddzielone Normalizacją Wsadu (`BatchNorm2d`) i funkcją aktywacji ReLU, zakończone redukcją wymiaru (`MaxPool2d`).
* **Blok 3:** Pojedyncza warstwa konwolucyjna z BatchNorm, ReLU i MaxPool.
* **Klasyfikator:** W pełni połączone warstwy liniowe (`Linear`) redukujące wektor z 512 do 37 klas, zabezpieczone warstwą `Dropout` w celu zapobiegania przeuczeniu.

## Eksperymenty i Wyniki

Przeprowadzono trzy eksperymenty, trenując sieć przez 20 epok z różnymi hiperparametrami, aby zaobserwować zjawisko overfittingu (przeuczenia) oraz sposoby walki z nim.

| Nazwa | BatchNorm | Dropout | Augmentacja | Max Train Acc | Max Test Acc | Wnioski |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Basic** | ❌ | 0.0 | ❌ | ~99.9% | ~12.0% | Książkowy overfitting. Model zapamiętał zbiór treningowy, całkowicie oblewając na zbiorze testowym. |
| **Regularization** | ✅ | 0.3 | ❌ | ~75.7% | ~21.3% | Dodanie BatchNorm i Dropout spowolniło przeuczanie, pozwalając modelowi nauczyć się bardziej ogólnych cech. |
| **Maxed Out** | ✅ | 0.5 | ✅ | ~54.1% | **~26.8%** | **Zwycięzca.** Dzięki augmentacji (obroty, odbicia) model widział zróżnicowane dane. Uczył się najwolniej (tylko 54% na treningu), ale osiągnął najwyższy, wciąż rosnący wynik na teście. |

### Wizualizacja Wyników
*(W tym miejscu w repozytorium GitHub możesz wstawić wygenerowany przez kod wykres `Comparing Experiments - Test Accuracy`)*
Na wykresie wyraźnie widać, że linia "Maxed Out" zachowuje zdrowy, rosnący trend po 20 epokach, podczas gdy pozostałe konfiguracje osiągają plateau (zatrzymują się).

## Sugestie na przyszłość (Co poprawić?)

Obecny wynik testowy to ok. 27%. Zadanie "Fine-grained classification" na 37 klasach jest bardzo trudne dla płytkiej sieci trenowanej od zera. Aby osiągnąć wyniki rzędu 80-90%, rekomendowane są następujące kroki:

1. **Dłuższy trening:** Krzywa uczenia dla "Maxed Out" w 20 epoce wciąż rośnie. Zwiększenie liczby epok do 50-100 prawdopodobnie poprawi dokładność.
2. **Transfer Learning:** Zastąpienie `PetNet` gotową, głęboką architekturą (np. `ResNet18` lub `EfficientNet`), która została wcześniej wstępnie wytrenowana na ogromnym zbiorze ImageNet. To branżowy standard dla małych zbiorów danych.
3. **Zwiększenie rozdzielczości obrazu:** Przejście z wymiarów `128x128` na `224x224`. Różnice między rasami zwierząt tkwią w detalach, które "gubią się" przy niskiej rozdzielczości.
4. **Learning Rate Scheduler:** Zastosowanie mechanizmu, który automatycznie zmniejsza krok uczenia (Learning Rate), gdy model przestaje robić postępy, aby "dostroić" wagi w końcowej fazie treningu.
5. **Weight Decay (L2 Regularization):** Dodanie kary za duże wagi do optymalizatora (np. `weight_decay=1e-4` w Adam optimizer), co wymusi jeszcze łagodniejszą formę krzywych.