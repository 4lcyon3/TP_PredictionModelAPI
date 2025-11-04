# train_model_hybrid.py
import keras
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
import os

def train(csv_path="dataset_3ro_clean.csv", out_dir="models_hybrid", label_auto_threshold=None):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # Ensure required cols exist
    required = ["persistente", "competente", "observador", "score_total"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")
    # If final_performance is empty, generate labels using threshold (default: global mean)
    if "final_performance" not in df.columns or df["final_performance"].isnull().all() or (df["final_performance"].astype(str).str.strip()=="").all():
        if label_auto_threshold is None:
            label_auto_threshold = df["score_total"].mean()
        df["final_performance"] = np.where(df["score_total"] >= label_auto_threshold, "aprobará", "no_aprobará")
        print("Se generaron etiquetas automáticamente con umbral:", label_auto_threshold)
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["final_performance"].astype(str))
    # Features: the three capacities + score_total
    X = df[["persistente","competente","observador","score_total"]].astype(float).values
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_s, y_train)
    # TensorFlow model (binary)
    n_features = X_train_s.shape[1]
    keras.backend.clear_session()
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train, validation_data=(X_test_s, y_test), epochs=100, batch_size=8,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)], verbose=2)
    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(out_dir, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    model.save(os.path.join(out_dir, "tf_model.keras"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))
    print("Modelos guardados en", out_dir)
    # Metrics quick
    pred_rf = rf.predict_proba(X_test_s)[:,1]
    pred_tf = model.predict(X_test_s).reshape(-1)
    pred_ens = (pred_rf + pred_tf)/2.0
    y_prob = pred_ens
    y_pred = (y_prob >= 0.5).astype(int)
    acc = (y_pred == y_test).mean()
    print("Accuracy ensemble (threshold 0.5):", acc)

if __name__ == "__main__":
    train()
