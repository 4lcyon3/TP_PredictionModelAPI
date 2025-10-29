import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def entrenar_modelo(ruta_csv="dataset_entrenamiento_final.csv"):
    df = pd.read_csv(ruta_csv)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["final_performance"])

    X = df[["score_total"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=30, batch_size=4)

    os.makedirs("modelos", exist_ok=True)
    joblib.dump(rf, "modelos/random_forest.pkl")
    joblib.dump(scaler, "modelos/scaler.pkl")
    model.save("modelos/tf_model.h5")
    joblib.dump(le, "modelos/label_encoder.pkl")

    acc = rf.score(X_test_scaled, y_test)
    print(f"Modelo entrenado. Accuracy RF: {acc:.2f}")

def predecir(score_total):
    rf = joblib.load("modelos/random_forest.pkl")
    scaler = joblib.load("modelos/scaler.pkl")
    le = joblib.load("modelos/label_encoder.pkl")

    X_input = scaler.transform(np.array([[score_total]]))
    pred_rf = rf.predict_proba(X_input)[0][1]

    model = keras.saving.load_model("modelos/tf_model.h5")
    pred_tf = model.predict(X_input)[0][0]

    # Promedio entre ambos modelos
    prob_final = (pred_rf + pred_tf) / 2
    label_pred = le.inverse_transform([int(prob_final >= 0.5)])[0]

    return {"probabilidad": float(prob_final), "rendimiento_predicho": label_pred}
