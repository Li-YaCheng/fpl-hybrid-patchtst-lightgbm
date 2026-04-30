from __future__ import annotations

"""
Paper-replica PatchTST-lite (Stage-1) as described in ACM双栏.pdf.

Key specs (from the paper):
- Input: X in R^{L x d}, L=30, d=20
- Patch embedding: Conv1D with patch_len=5 (so N=6 patches), stride=patch_len
- Learnable positional encoding (per patch)
- Transformer encoder: 2 layers, 4 heads, FF dim = 128
- Pooling: Global Average Pooling over patches
- Output: sigmoid probability p
- Loss: focal loss (alpha=0.75, gamma=1.8)
- Optim: Adam lr=1e-3, L2=1e-4, dropout=0.50
- Scheduler: ReduceLROnPlateau (factor=0.5, min_lr=1e-5)
- EarlyStopping: patience=8, restore best
- Multi-seed ensemble: seeds {42,43,44} -> mu_p, sigma_p
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchTSTPaperConfig:
    seq_len: int = 30
    n_features: int = 20
    patch_len: int = 5
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.50
    l2: float = 1e-4
    lr: float = 1e-3
    alpha: float = 0.75
    gamma: float = 1.8
    batch_size: int = 256
    max_epochs: int = 60
    early_patience: int = 8
    rlrop_factor: float = 0.5
    rlrop_patience: int = 3
    min_lr: float = 1e-5


def _set_seeds(seed: int) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf  # type: ignore

    tf.random.set_seed(seed)


def focal_loss(alpha: float, gamma: float):
    import tensorflow as tf  # type: ignore

    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        # p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        loss = -alpha_t * tf.pow((1.0 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return _loss


def build_patchtst_paper(cfg: PatchTSTPaperConfig):
    import tensorflow as tf  # type: ignore

    if cfg.seq_len % cfg.patch_len != 0:
        raise ValueError("seq_len must be divisible by patch_len")
    n_patches = cfg.seq_len // cfg.patch_len

    reg = tf.keras.regularizers.l2(cfg.l2)

    inp = tf.keras.Input(shape=(cfg.seq_len, cfg.n_features), name="x")

    # Conv1D patch embedding
    x = tf.keras.layers.Conv1D(
        filters=cfg.d_model,
        kernel_size=cfg.patch_len,
        strides=cfg.patch_len,
        padding="valid",
        kernel_regularizer=reg,
        name="patch_conv",
    )(inp)  # (B, N, d_model)

    # learnable positional encoding
    pos = tf.keras.layers.Embedding(input_dim=n_patches, output_dim=cfg.d_model, name="pos_emb")
    idx = tf.range(n_patches)
    x = x + pos(idx)[None, :, :]

    for i in range(cfg.n_layers):
        h = tf.keras.layers.LayerNormalization(name=f"ln1_{i}")(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.n_heads,
            key_dim=cfg.d_model // cfg.n_heads,
            dropout=cfg.dropout,
            name=f"mha_{i}",
        )(h, h)
        attn = tf.keras.layers.Dropout(cfg.dropout, name=f"drop_attn_{i}")(attn)
        x = tf.keras.layers.Add(name=f"res1_{i}")([x, attn])

        h = tf.keras.layers.LayerNormalization(name=f"ln2_{i}")(x)
        ff = tf.keras.layers.Dense(cfg.ff_dim, activation="gelu", kernel_regularizer=reg, name=f"ff1_{i}")(h)
        ff = tf.keras.layers.Dropout(cfg.dropout, name=f"drop_ff_{i}")(ff)
        ff = tf.keras.layers.Dense(cfg.d_model, kernel_regularizer=reg, name=f"ff2_{i}")(ff)
        ff = tf.keras.layers.Dropout(cfg.dropout, name=f"drop_ff2_{i}")(ff)
        x = tf.keras.layers.Add(name=f"res2_{i}")([x, ff])

    x = tf.keras.layers.LayerNormalization(name="ln_out")(x)
    emb = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)  # embedding
    emb = tf.keras.layers.Dropout(cfg.dropout, name="emb_drop")(emb)
    p = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="p")(emb)

    model = tf.keras.Model(inputs=inp, outputs=[p, emb], name="patchtst_paper")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss={"p": focal_loss(cfg.alpha, cfg.gamma)},
        metrics={"p": [tf.keras.metrics.AUC(name="auc")]},
    )
    return model


def fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    cfg: PatchTSTPaperConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns p_val, emb_val, p_test, emb_test.
    """
    import tensorflow as tf  # type: ignore

    _set_seeds(seed)
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    X_te = X_te.astype(np.float32)
    y_tr = y_tr.astype(np.float32).reshape(-1, 1)
    y_va = y_va.astype(np.float32).reshape(-1, 1)

    model = build_patchtst_paper(cfg)
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_p_auc",
            mode="max",
            patience=int(cfg.early_patience),
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_p_auc",
            mode="max",
            factor=float(cfg.rlrop_factor),
            patience=int(cfg.rlrop_patience),
            min_lr=float(cfg.min_lr),
            verbose=0,
        ),
    ]

    model.fit(
        X_tr,
        {"p": y_tr},
        validation_data=(X_va, {"p": y_va}),
        epochs=int(cfg.max_epochs),
        batch_size=int(cfg.batch_size),
        verbose=0,
        callbacks=cbs,
    )

    p_va, emb_va = model.predict(X_va, batch_size=int(cfg.batch_size), verbose=0)
    p_te, emb_te = model.predict(X_te, batch_size=int(cfg.batch_size), verbose=0)
    return p_va.reshape(-1), emb_va, p_te.reshape(-1), emb_te


def fit_predict_all(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    seed: int,
    cfg: PatchTSTPaperConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns p_train, emb_train, p_val, emb_val, p_test, emb_test.
    """
    import tensorflow as tf  # type: ignore

    _set_seeds(seed)
    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    X_te = X_te.astype(np.float32)
    y_tr = y_tr.astype(np.float32).reshape(-1, 1)
    y_va = y_va.astype(np.float32).reshape(-1, 1)

    model = build_patchtst_paper(cfg)
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_p_auc",
            mode="max",
            patience=int(cfg.early_patience),
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_p_auc",
            mode="max",
            factor=float(cfg.rlrop_factor),
            patience=int(cfg.rlrop_patience),
            min_lr=float(cfg.min_lr),
            verbose=0,
        ),
    ]

    model.fit(
        X_tr,
        {"p": y_tr},
        validation_data=(X_va, {"p": y_va}),
        epochs=int(cfg.max_epochs),
        batch_size=int(cfg.batch_size),
        verbose=0,
        callbacks=cbs,
    )

    p_tr, emb_tr = model.predict(X_tr, batch_size=int(cfg.batch_size), verbose=0)
    p_va, emb_va = model.predict(X_va, batch_size=int(cfg.batch_size), verbose=0)
    p_te, emb_te = model.predict(X_te, batch_size=int(cfg.batch_size), verbose=0)
    return (
        p_tr.reshape(-1),
        emb_tr,
        p_va.reshape(-1),
        emb_va,
        p_te.reshape(-1),
        emb_te,
    )


def ensemble_mu_sigma(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    *,
    seeds: list[int],
    cfg: PatchTSTPaperConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train K models (different seeds) and return:
      mu_val, sig_val, mu_test, sig_test
    """
    pvals = []
    ptests = []
    for s in seeds:
        p_va, _e_va, p_te, _e_te = fit_predict(X_tr, y_tr, X_va, y_va, X_te, seed=int(s), cfg=cfg)
        pvals.append(p_va)
        ptests.append(p_te)
    Pva = np.stack(pvals, axis=0)  # (K,Nv)
    Pte = np.stack(ptests, axis=0)
    mu_va = Pva.mean(axis=0)
    sig_va = Pva.std(axis=0, ddof=0)
    mu_te = Pte.mean(axis=0)
    sig_te = Pte.std(axis=0, ddof=0)
    return mu_va, sig_va, mu_te, sig_te

