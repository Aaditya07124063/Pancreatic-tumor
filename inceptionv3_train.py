"""
InceptionV3 - Pancreatic CT Binary Classification
Fine-tuned with 5-seed evaluation | 80/10/10 stratified split
Image size: 299x299 (InceptionV3 native)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             cohen_kappa_score, confusion_matrix)

from data_utils import (CLASSES, SEEDS, EPOCHS, PATIENCE,
                        load_merged_dataset, stratified_split, print_split_summary)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE   = (299, 299)
BATCH_SIZE = 32
OUTPUT_DIR = "./inceptionv3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_images(paths, img_size):
    imgs = []
    for p in paths:
        img = load_img(p, target_size=img_size)
        arr = img_to_array(img)
        imgs.append(arr)
    return np.array(imgs, dtype="float32")


# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.rot90(image, k=tf.random.uniform(
        shape=[], minval=0, maxval=4, dtype=tf.int32))
    boxes      = tf.random.uniform(shape=[1, 4], minval=0.0, maxval=0.1)
    box_coords = [boxes[0, 0], boxes[0, 1],
                  1.0 - boxes[0, 2], 1.0 - boxes[0, 3]]
    image = tf.image.crop_and_resize(
        tf.expand_dims(image, 0),
        [box_coords], [0], IMG_SIZE
    )[0]
    return image


def make_dataset(paths, labels, img_size, augment_flag, batch_size, shuffle=False, seed=42):
    images = load_images(paths, img_size)
    images = preprocess_input(images)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if augment_flag:
        ds = ds.map(lambda x, y: (augment(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── MODEL ────────────────────────────────────────────────────────────────────
def build_model():
    base = InceptionV3(weights="imagenet", include_top=False,
                       input_shape=(*IMG_SIZE, 3))
    base.trainable = True

    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x   = base(inp, training=True)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── SET SEED ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─── TRAIN ONE SEED ───────────────────────────────────────────────────────────
def train_one_seed(seed, all_paths, all_labels):
    set_seed(seed)

    tr_p, va_p, te_p, tr_l, va_l, te_l = stratified_split(all_paths, all_labels, seed)
    print_split_summary(tr_l, va_l, te_l)

    train_ds = make_dataset(tr_p, tr_l, IMG_SIZE, True,  BATCH_SIZE, shuffle=True, seed=seed)
    val_ds   = make_dataset(va_p, va_l, IMG_SIZE, False, BATCH_SIZE, shuffle=False)
    test_ds  = make_dataset(te_p, te_l, IMG_SIZE, False, BATCH_SIZE, shuffle=False)

    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, verbose=0)
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1
    )

    train_loss, train_acc = model.evaluate(
        make_dataset(tr_p, tr_l, IMG_SIZE, False, BATCH_SIZE, shuffle=False), verbose=0)
    val_loss,   val_acc   = model.evaluate(val_ds,  verbose=0)
    test_loss,  test_acc  = model.evaluate(test_ds, verbose=0)

    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0).ravel()
        y_true.extend(batch_y.numpy().astype(int))
        y_pred.extend((probs >= 0.5).astype(int))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    model.save(os.path.join(OUTPUT_DIR, f"inceptionv3_seed{seed}.keras"))

    return {
        "seed": seed, "train_acc": train_acc, "val_acc": val_acc,
        "test_acc": test_acc, "test_loss": test_loss,
        "precision": prec, "recall": rec, "f1": f1,
        "kappa": kappa, "cm": cm, "history": history,
        "y_true": y_true, "y_pred": y_pred,
    }


# ─── PLOTS ────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, seed, class_names=CLASSES):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set(xticks=list(ticks), yticks=list(ticks),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label",
           title=f"InceptionV3 Confusion Matrix (seed={seed})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_seed{seed}.png"), dpi=150)
    plt.close()


def plot_training_curves(history, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set(title=f"Accuracy (seed={seed})", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()
    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set(title=f"Loss (seed={seed})", xlabel="Epoch", ylabel="Loss")
    ax2.legend()
    plt.suptitle(f"InceptionV3 Training Curves — seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_seed{seed}.png"), dpi=150)
    plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  InceptionV3 — Pancreatic CT Classification  |  5-Seed Eval")
    print("=" * 65)

    all_paths, all_labels = load_merged_dataset(dedupe=True)
    print(f"Total deduplicated images: {len(all_paths)}  |  Classes: {CLASSES}\n")

    results = []
    for seed in SEEDS:
        print(f"\n{'─'*55}")
        print(f"  Training seed={seed}")
        print(f"{'─'*55}")
        res = train_one_seed(seed, all_paths, all_labels)
        results.append(res)
        plot_confusion_matrix(res["cm"], seed)
        plot_training_curves(res["history"], seed)
        print(f"  seed={seed} | test_acc={res['test_acc']:.4f} | "
              f"F1={res['f1']:.4f} | Kappa={res['kappa']:.4f}")

    cols = ["train_acc", "val_acc", "test_acc", "precision",
            "recall", "f1", "kappa", "test_loss"]

    print("\n" + "=" * 90)
    print(f"  {'Seed':<6} {'Train Acc':>10} {'Val Acc':>9} {'Test Acc':>9} "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8} {'Kappa':>8} {'Loss':>8}")
    print("=" * 90)
    for r in results:
        print(f"  {r['seed']:<6} {r['train_acc']:>10.4f} {r['val_acc']:>9.4f} "
              f"{r['test_acc']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['kappa']:>8.4f} {r['test_loss']:>8.4f}")

    avgs = {c: np.mean([r[c] for r in results]) for c in cols}
    print("─" * 90)
    print(f"  {'AVG':<6} {avgs['train_acc']:>10.4f} {avgs['val_acc']:>9.4f} "
          f"{avgs['test_acc']:>9.4f} {avgs['precision']:>10.4f} {avgs['recall']:>8.4f} "
          f"{avgs['f1']:>8.4f} {avgs['kappa']:>8.4f} {avgs['test_loss']:>8.4f}")
    print("=" * 90)

    rows = [{k: r[k] for k in cols + ["seed"]} for r in results]
    avg_row = {k: avgs[k] for k in cols}
    avg_row["seed"] = "AVG"
    rows.append(avg_row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "inceptionv3_results.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()