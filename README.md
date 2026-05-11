# Task_3 – Klasyfikacja obrazów CIFAR-10 w PyTorch

Projekt został przygotowany w ramach zadania dotyczącego implementacji sieci neuronowej w bibliotece PyTorch dla zbioru CIFAR-10.

## Cel zadania

Celem projektu było zbudowanie i przetestowanie modelu sieci neuronowej do klasyfikacji obrazów ze zbioru CIFAR-10 bez korzystania z gotowych modeli pretrained.  
Ze względu na ostatnią cyfrę indeksu 0 w projekcie zaimplementowano sieć typu MLP (Multilayer Perceptron).

## Zbiór danych

W projekcie wykorzystano zbiór CIFAR-10 zawierający 60 000 kolorowych obrazów o rozmiarze 32x32 piksele, podzielonych na 10 klas:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Zbiór został pobrany z użyciem `torchvision.datasets.CIFAR10`.

## Zakres projektu

W projekcie wykonano:
- analizę zbioru danych,
- wizualizację przykładowych obrazów,
- obliczenie podstawowych statystyk danych,
- implementację własnej architektury MLP,
- trening i ewaluację modelu,
- analizę accuracy,
- confusion matrix,
- classification report,
- wizualizację krzywych uczenia,
- analizę błędnie sklasyfikowanych obrazów,
- serię eksperymentów dla różnych hiperparametrów.

## Eksperymenty

Przeprowadzono eksperymenty badające wpływ:
- learning rate,
- batch size,
- liczby neuronów w warstwach ukrytych,
- dropout.

Dla każdej serii eksperymentów przedstawiono tabelę wyników, wykresy oraz krótkie wnioski.

## Struktura plików

- `report.ipynb` – główne sprawozdanie w formie notebooka Jupyter
- `data.py` – przygotowanie danych i analiza zbioru
- `models.py` – definicja modelu MLP
- `train.py` – trening modelu
- `evaluate.py` – metryki i ewaluacja
- `plots.py` – funkcje do wizualizacji
- `requirements.txt` – lista wymaganych bibliotek

## Uruchomienie projektu

1. Zainstalować wymagane biblioteki:
```bash
pip install -r requirements.txt
```

2. Uruchomić notebook:
```bash
jupyter notebook
```

3. Otworzyć plik `report.ipynb` i wykonać wszystkie komórki.

## Wymagane biblioteki

Projekt korzysta między innymi z bibliotek:
- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

## Autor

Projekt wykonany w ramach zadania **Task_3**.
