# Wybierz obraz bazowy
FROM python:3.8-slim

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj plik requirements.txt do kontenera
COPY requirements.txt .

# Zainstaluj zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt