"""
evaluate_external.py
====================
Evaluates the saved ResNet50 / InceptionV3 checkpoints on the held-out ./test
folder (412 images) that the end-to-end fine-tuning runs never touched.

Why this matters
----------------
resnet50_train.py, inceptionv3_train.py, mobilevit_train.py and
swin_transformer_train.py all set DATA_DIR = "./train" and then carve an
80/10/10 split out of that folder alone. The ./test folder was only ever used by
feature_extraction_pipeline.py. So ./test is a genuine external hold-out for the
four fine-tuned networks.

It is also a *harder* and *cleaner* hold-out:
  - the internal 100-image test split shares 29-36 byte-identical images with its
    own training fold (duplicate files inside ./train)
  - ./test shares only 8 byte-identical images with ./train

The script reports metrics on the full ./test folder and, separately, on ./test
with those 8 leaked images removed.

Usage:
    python evaluate_external.py
"""

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inception

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

CLASSES = ["normal", "pancreatic_tumor"]
SEEDS = [42, 7, 21, 99, 123]

MODEL_SPECS = {
    "resnet50": dict(img_size=(224, 224), preprocess=pp_resnet,
                     outdir="./resnet50_outputs", ckpt="resnet50_seed{seed}.keras",
                     pretty="ResNet50"),
    "inceptionv3": dict(img_size=(299, 299), preprocess=pp_inception,
                        outdir="./inceptionv3_outputs", ckpt="inceptionv3_seed{seed}.keras",
                        pretty="InceptionV3"),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(data_dir):
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(data_dir, cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(d, f))
                labels.append(label)
    return np.array(paths), np.array(labels)


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        n=len(y_true),
        acc=accuracy_score(y_true, y_pred),
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        f1=f1_score(y_true, y_pred, zero_division=0),
        kappa=cohen_kappa_score(y_true, y_pred),
        auroc=roc_auc_score(y_true, y_prob),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def main():
    te_paths, te_labels = load_dataset("./test")
    tr_paths, _ = load_dataset("./train")

    train_hashes = {md5(p) for p in tr_paths}
    leaked = np.array([md5(p) in train_hashes for p in te_paths])

    print("=" * 108)
    print("  EXTERNAL HOLD-OUT EVALUATION — ./test folder (never seen by the fine-tuning runs)")
    print("=" * 108)
    print(f"  ./test : {len(te_paths)} images  "
          f"({int((te_labels == 0).sum())} normal / {int(te_labels.sum())} tumour)")
    print(f"  byte-identical to a ./train image: {int(leaked.sum())}  -> also reporting a de-duplicated subset\n")

    all_rows = []
    for name, spec in MODEL_SPECS.items():
        print(f"\n### {spec['pretty']} ###")
        X = spec["preprocess"](
            np.array([img_to_array(load_img(p, target_size=spec["img_size"]))
                      for p in te_paths], dtype="float32"))

        for seed in SEEDS:
            ckpt = os.path.join(spec["outdir"], spec["ckpt"].format(seed=seed))
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt}")
                continue
            model = tf.keras.models.load_model(ckpt, compile=False)
            prob = model.predict(X, batch_size=32, verbose=0).ravel()
            pred = (prob >= 0.5).astype(int)

            full = metrics(te_labels, pred, prob)
            clean = metrics(te_labels[~leaked], pred[~leaked], prob[~leaked])

            all_rows.append(dict(model=spec["pretty"], seed=seed, subset="full", **full))
            all_rows.append(dict(model=spec["pretty"], seed=seed, subset="dedup", **clean))
            print(f"  seed {seed:>4} | full  acc={full['acc']:.4f} bal={full['balanced_acc']:.4f} "
                  f"F1={full['f1']:.4f} kappa={full['kappa']:.4f} auroc={full['auroc']:.4f} "
                  f"TN/FP/FN/TP={full['tn']}/{full['fp']}/{full['fn']}/{full['tp']}")
            print(f"           | dedup acc={clean['acc']:.4f} bal={clean['balanced_acc']:.4f} "
                  f"F1={clean['f1']:.4f} kappa={clean['kappa']:.4f} auroc={clean['auroc']:.4f}")
            del model

    df = pd.DataFrame(all_rows)
    df.to_csv("./external_test_results.csv", index=False)

    print("\n" + "=" * 108)
    print("  SUMMARY — mean +/- SD over 5 seeds, external ./test hold-out")
    print("=" * 108)
    cols = ["acc", "balanced_acc", "precision", "recall", "specificity", "f1", "kappa", "auroc"]
    hdr = f"  {'Model':<14}{'Subset':<8}" + "".join(f"{c[:9]:>11}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for (m, s), g in df.groupby(["model", "subset"], sort=False):
        print(f"  {m:<14}{s:<8}" +
              "".join(f"{g[c].mean():>6.4f}±{g[c].std(ddof=0):<4.3f}" for c in cols))
    print("=" * len(hdr))
    print("\n  Saved -> ./external_test_results.csv")


if __name__ == "__main__":
    main()
