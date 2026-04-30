from __future__ import annotations

"""
Lightweight PatchTST-like encoder (reconstructed).

This is NOT a faithful reproduction of the original PatchTST implementation.
It is a small, self-contained temporal encoder that provides:
- a probability head p(y=1|x)
- an embedding vector per sequence (for "emb" variants)

It is designed to be:
- reproducible (seeded)
- reasonably fast
- dependency-light (TensorFlow/Keras only)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchTSTConfig:
    seq_len: int
    d_model: int = 64
    n_heads: int = 4
    ff_mult: int = 2
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 2e-3
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5


def _set_seeds(seed: int) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass


def _build_model(seq_len: int, n_features: int, cfg: PatchTSTConfig):
    import tensorflow as tf  # type: ignore

    inputs = tf.keras.Input(shape=(seq_len, n_features), name="x")

    # Project to model dim
    x = tf.keras.layers.Dense(cfg.d_model, name="proj")(inputs)

    for i in range(cfg.n_layers):
        # Pre-norm self-attention block
        h = tf.keras.layers.LayerNormalization(name=f"ln1_{i}")(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.n_heads,
            key_dim=max(8, cfg.d_model // cfg.n_heads),
            dropout=cfg.dropout,
            name=f"mha_{i}",
        )(h, h)
        x = tf.keras.layers.Add(name=f"res1_{i}")([x, attn])

        h = tf.keras.layers.LayerNormalization(name=f"ln2_{i}")(x)
        ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.d_model * cfg.ff_mult, activation="gelu"),
                tf.keras.layers.Dropout(cfg.dropout),
                tf.keras.layers.Dense(cfg.d_model),
            ],
            name=f"ff_{i}",
        )(h)
        x = tf.keras.layers.Add(name=f"res2_{i}")([x, ff])

    # Pool to embedding
    h = tf.keras.layers.LayerNormalization(name="ln_out")(x)
    emb = tf.keras.layers.GlobalAveragePooling1D(name="emb")(h)
    emb = tf.keras.layers.Dropout(cfg.dropout, name="emb_drop")(emb)
    emb_out = tf.keras.layers.Lambda(lambda t: t, name="emb_out")(emb)

    # Prob head
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="p")(emb_out)

    model = tf.keras.Model(inputs=inputs, outputs=[p, emb_out], name="patchtst_light")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss={"p": "binary_crossentropy"},
        metrics={"p": [tf.keras.metrics.AUC(name="auc")]},
    )
    return model


def fit_predict_embed(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    cfg: PatchTSTConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      p_val, emb_val, p_test, emb_test
    """
    import tensorflow as tf  # type: ignore

    _set_seeds(seed)

    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    X_te = X_te.astype(np.float32)
    y_tr = y_tr.astype(np.float32).reshape(-1, 1)
    y_va = y_va.astype(np.float32).reshape(-1, 1)

    model = _build_model(X_tr.shape[1], X_tr.shape[2], cfg)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_p_auc",
            mode="max",
            patience=int(cfg.patience),
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_tr,
        {"p": y_tr},
        validation_data=(X_va, {"p": y_va}),
        epochs=int(cfg.max_epochs),
        batch_size=int(cfg.batch_size),
        verbose=0,
        callbacks=cb,
    )

    p_va, emb_va = model.predict(X_va, batch_size=int(cfg.batch_size), verbose=0)
    p_te, emb_te = model.predict(X_te, batch_size=int(cfg.batch_size), verbose=0)
    return p_va.reshape(-1), emb_va, p_te.reshape(-1), emb_te

