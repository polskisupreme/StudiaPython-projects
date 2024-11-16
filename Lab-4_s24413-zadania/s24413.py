# s24413.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generowanie prostego zbioru danych
def generate_data():
    np.random.seed(42)

    # Generate first cluster centered at (2, 2)
    cluster_1 = np.random.normal(loc=(2, 2), scale=0.5, size=(100, 2))
    labels_1 = np.zeros(100)

    # Generate second cluster centered at (6, 6)
    cluster_2 = np.random.normal(loc=(6, 6), scale=0.5, size=(100, 2))
    labels_2 = np.ones(100)

    # Combine clusters
    data = np.vstack((cluster_1, cluster_2))
    labels = np.hstack((labels_1, labels_2))

    return data, labels

# Trenowanie prostego modelu regresji logistycznej
def train_model():
    data, labels = generate_data()

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)

    # Wyliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)

    # Zapis wyniku
    with open("accuracy.txt", "w") as f:
        f.write(f"Model trained with accuracy: {accuracy * 100:.2f}%\n")

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
