"""Experimento de pronóstico multi-horizonte con atención y Transformer.

La señal se genera de forma reproducible y no representa mediciones reales.
El script entrena los modelos en TensorFlow/Keras y guarda métricas, figuras,
predicciones y pesos de atención en resultados/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "resultados"
OUT.mkdir(parents=True, exist_ok=True)
N = 1008                 # seis semanas de observaciones horarias
LOOKBACK = 48            # dos días de historia
HORIZON = 6              # pronóstico de las siguientes seis horas
SEEDS = (7, 23, 58)


def make_series(n: int = N, seed: int = 314) -> np.ndarray:
    """Serie horaria didáctica con patrón diario, semanal y un cambio suave."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    hour = 2 * np.pi * t / 24
    week = 2 * np.pi * t / 168
    weekend = ((t.astype(int) // 24) % 7 >= 5).astype(np.float32)
    transition = 1.0 / (1.0 + np.exp(-(t - 670.0) / 18.0))
    level = 0.0015 * t + 0.42 * transition
    daily = (0.72 + 0.10 * transition) * np.sin(hour - 0.8)
    weekly = 0.38 * np.sin(week + 0.5)
    workday = 0.22 * np.sin(2 * hour + 0.3) * (1.0 - 0.55 * weekend)
    noise = rng.normal(0, 0.13, size=n)
    return (2.5 + level + daily + weekly + workday + noise).astype(np.float32)


def make_windows(values: np.ndarray):
    cut = int(0.80 * len(values))
    mean, std = float(values[:cut].mean()), float(values[:cut].std())
    scaled = (values - mean) / (std + 1e-8)
    xs, ys, starts = [], [], []
    for start in range(LOOKBACK, len(values) - HORIZON + 1, HORIZON):
        xs.append(scaled[start - LOOKBACK:start])
        ys.append(scaled[start:start + HORIZON])
        starts.append(start)
    x = np.asarray(xs, dtype=np.float32)[..., None]
    y = np.asarray(ys, dtype=np.float32)
    starts = np.asarray(starts)
    train = np.flatnonzero(starts + HORIZON <= cut)
    test = np.flatnonzero(starts >= cut)
    val_start = int(0.85 * len(train))
    fit, val = train[:val_start], train[val_start:]
    return x, y, starts, fit, val, test, mean, std, cut, scaled


class AdditiveTemporalAttention(layers.Layer):
    """Pondera posiciones pasadas usando el estado final como consulta."""
    def __init__(self, width: int, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1, use_bias=False, name="puntuacion")

    def call(self, x):
        query = x[:, -1:, :]
        weights = tf.nn.softmax(self.score(tf.tanh(x + query)), axis=1)
        context = tf.reduce_sum(weights * x, axis=1)
        return context, tf.squeeze(weights, axis=-1)


class PositionEmbedding(layers.Layer):
    def __init__(self, length: int, width: int, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.embedding = layers.Embedding(length, width, name="posiciones")

    def call(self, x):
        ids = tf.range(self.length)
        return x + self.embedding(ids)[None, :, :]


def build_attention_model():
    inp = keras.Input((LOOKBACK, 1), name="historia_48_horas")
    h = layers.Dense(32, activation="tanh", name="proyeccion_temporal")(inp)
    context, weights = AdditiveTemporalAttention(32, name="atencion_aditiva")(h)
    out = layers.Dense(HORIZON, name="seis_horas_pronosticadas")(context)
    return keras.Model(inp, out), keras.Model(inp, weights)


def build_transformer_model():
    inp = keras.Input((LOOKBACK, 1), name="historia_48_horas")
    x = layers.Dense(32, name="embedding_valor")(inp)
    x = PositionEmbedding(LOOKBACK, 32, name="embedding_posicional")(x)
    a = layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.10,
                                  name="autoatencion")(x, x, use_causal_mask=True)
    x = layers.LayerNormalization(epsilon=1e-6, name="normalizacion_atencion")(x + a)
    f = layers.Dense(64, activation="gelu", name="feed_forward_1")(x)
    f = layers.Dense(32, name="feed_forward_2")(f)
    x = layers.LayerNormalization(epsilon=1e-6, name="normalizacion_feed_forward")(x + f)
    out = layers.Dense(HORIZON, name="pronostico_6_horas")(x[:, -1, :])
    return keras.Model(inp, out, name="transformer_causal")


def fit_model(builder, x, y, fit_ids, val_ids, seed):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model, attention = builder() if builder is build_attention_model else (builder(), None)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True)
    history = model.fit(x[fit_ids], y[fit_ids], validation_data=(x[val_ids], y[val_ids]),
                        epochs=50, batch_size=32, shuffle=False, verbose=0,
                        callbacks=[stop])
    return model, attention, history


def metrics(true, pred):
    return {"MAE": float(np.mean(np.abs(true - pred))),
            "RMSE": float(np.sqrt(np.mean((true - pred) ** 2)))}


def main():
    values = make_series()
    x, y, starts, fit_ids, val_ids, test_ids, mean, std, cut, scaled = make_windows(values)
    truth_scaled = y[test_ids]
    truth = truth_scaled * std + mean
    rows, pred_rows, last_attention = [], [], None
    for name, builder in [("Atención aditiva", build_attention_model),
                          ("Transformer", build_transformer_model)]:
        for seed in SEEDS:
            model, attention, history = fit_model(builder, x, y, fit_ids, val_ids, seed)
            pred = model.predict(x[test_ids], verbose=0) * std + mean
            row = {"modelo": name, "semilla": seed, **metrics(truth, pred),
                   "epocas": len(history.history["loss"])}
            rows.append(row)
            if name == "Atención aditiva" and seed == SEEDS[0]:
                last_attention = attention.predict(x[test_ids[:1]], verbose=0)[0]
        
    # Referencias sin entrenamiento: último valor observado y repetición del
    # mismo tramo horario del día anterior (disponible dentro de la ventana).
    last = np.repeat(x[test_ids, -1, 0, None] * std + mean, HORIZON, axis=1)
    daily = x[test_ids, -24:-24 + HORIZON, 0] * std + mean
    baseline_rows = [
        {"modelo": "Persistencia", "semilla": "—", **metrics(truth, last), "epocas": 0},
        {"modelo": "Estacional (24 h)", "semilla": "—", **metrics(truth, daily), "epocas": 0},
    ]
    pd.DataFrame(rows).to_csv(OUT / "metricas_semillas.csv", index=False)
    all_rows = baseline_rows + rows
    pd.DataFrame(all_rows).to_csv(OUT / "metricas_detalladas.csv", index=False)
    summary = {"datos": {"tipo": "señal sintética; no son mediciones reales", "puntos": N,
                         "lookback_horas": LOOKBACK, "horizonte_horas": HORIZON,
                         "corte": cut, "ventanas_prueba": int(len(test_ids)),
                         "semillas": list(SEEDS), "tensorflow": tf.__version__},
               "modelos": {}}
    for name in ["Persistencia", "Estacional (24 h)", "Atención aditiva", "Transformer"]:
        subset = [r for r in all_rows if r["modelo"] == name]
        summary["modelos"][name] = {
            "MAE_media": float(np.mean([r["MAE"] for r in subset])),
            "MAE_DE": float(np.std([r["MAE"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
            "RMSE_media": float(np.mean([r["RMSE"] for r in subset])),
            "RMSE_DE": float(np.std([r["RMSE"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
            "repeticiones": len(subset),
        }
    (OUT / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for i, row in enumerate(test_ids):
        record = {"inicio_prueba": int(starts[row])}
        for step in range(HORIZON):
            record[f"real_t+{step+1}"] = float(truth[i, step])
        pred_rows.append(record)
    pd.DataFrame(pred_rows).to_csv(OUT / "objetivos_prueba.csv", index=False)

    plt.figure(figsize=(10, 4.5))
    plt.plot(np.arange(N), values, color="#496b88", lw=1, label="Señal sintética")
    plt.axvline(cut, color="#c76d43", ls="--", label="Inicio del tramo de prueba")
    plt.xlabel("Hora simulada")
    plt.ylabel("Índice de demanda (unidades sintéticas)")
    plt.title("Serie didáctica y partición temporal")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "serie_particion.png", dpi=180)
    plt.close()

    grouped = pd.DataFrame(all_rows).groupby("modelo")["RMSE"].agg(["mean", "std"])
    names = ["Persistencia", "Estacional (24 h)", "Atención aditiva", "Transformer"]
    means = [grouped.loc[n, "mean"] for n in names]
    errors = [0 if np.isnan(grouped.loc[n, "std"]) else grouped.loc[n, "std"] for n in names]
    plt.figure(figsize=(8.5, 4.6))
    plt.bar(names, means, yerr=errors, capsize=4,
            color=["#9aa7af", "#7696a8", "#55a39a", "#d58861"])
    plt.ylabel("RMSE en escala original")
    plt.title("Error promedio en pronósticos de seis horas")
    plt.xticks(rotation=12, ha="right")
    plt.grid(axis="y", alpha=.22)
    plt.tight_layout()
    plt.savefig(OUT / "comparacion_rmse.png", dpi=180)
    plt.close()

    if last_attention is not None:
        plt.figure(figsize=(8.5, 3.8))
        plt.bar(np.arange(1, LOOKBACK + 1), last_attention, color="#55a39a")
        plt.xlabel("Posición de la hora en la ventana (1 = más antigua)")
        plt.ylabel("Peso de atención")
        plt.title("Pesos aprendidos por la atención aditiva")
        plt.tight_layout()
        plt.savefig(OUT / "pesos_atencion.png", dpi=180)
        plt.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
