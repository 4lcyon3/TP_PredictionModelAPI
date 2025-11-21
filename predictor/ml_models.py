import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import os

def train_all():
    data_path = "data/processed/dataset_limpio.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo {data_path}. Ejecuta data_preprocessing.py primero.")
    
    df = pd.read_csv(data_path)

    X = df[["persistente", "competente", "observador"]].values

    y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2) * np.random.uniform(0.8, 1.2, len(X))
    y= 100 * y / np.max(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=250, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    nn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    nn.fit(X_train_scaled, y_train, epochs=150, batch_size=8, verbose=0) # type: ignore

    nn_pred = nn.predict(X_test_scaled).flatten()

    hybrid_pred = (0.6 * rf_pred + 0.4 * nn_pred)

    rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
    r2 = r2_score(y_test, hybrid_pred)
    mae = mean_absolute_error(y_test, hybrid_pred)
    print(f"   Evaluación del modelo híbrido:")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R² = {r2:.4f}")
    print(f"   MAE = {mae:.4f}")

    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf, os.path.join(model_dir, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    nn.save(os.path.join(model_dir, "nn_model.h5"))

    return {"rmse": rmse, "r2": r2}

if __name__ == "__main__":
    train_all()