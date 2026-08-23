"""
CT Image Feature Extraction & Downstream Classification Pipeline
===============================================================
Backbones supported: Xception, DenseNet121
Classifiers trained: SVM, Random Forest, AdaBoost, KNN, XGBoost, Bagging, ANN, LSTM, Bi-LSTM
Features: 5-seed evaluation, automated plots (confusion matrices, curves), CSV reporting.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Import feature extractor networks
from tensorflow.keras.applications import Xception, DenseNet121
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet

# Classical & ANN classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)

# Evaluation metrics
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, recall_score, precision_score,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import data_utils as du

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configure TensorFlow for deterministic behavior if running on GPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def get_image_paths_labels(data_dir, classes):
    """Walks the directory and gathers image file paths and labels."""
    paths, labels = [], []
    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label_idx)
    return np.array(paths), np.array(labels)

def load_and_preprocess_images(paths, img_size, preprocess_fn):
    """Loads images from disk and applies backbone-specific preprocessing."""
    imgs = []
    print(f"  Loading {len(paths)} images...")
    for idx, path in enumerate(paths):
        img = load_img(path, target_size=img_size)
        arr = img_to_array(img)
        imgs.append(arr)
        if (idx + 1) % 100 == 0:
            print(f"    Loaded {idx + 1}/{len(paths)}...", flush=True)
    
    imgs = np.array(imgs, dtype="float32")
    imgs = preprocess_fn(imgs)
    return imgs

# ─── FEATURE EXTRACTOR ────────────────────────────────────────────────────────
def build_feature_extractor(backbone_name, img_size):
    """Initializes the specified feature extraction backbone."""
    input_shape = (*img_size, 3)
    if backbone_name.lower() == "xception":
        print("[INFO] Initializing Xception Backbone (pretrained on ImageNet)...")
        base = Xception(include_top=False, weights="imagenet",
                        input_shape=input_shape, pooling="max")
        preprocess_fn = preprocess_xception
    elif backbone_name.lower() == "densenet121":
        print("[INFO] Initializing DenseNet121 Backbone (pretrained on ImageNet)...")
        base = DenseNet121(include_top=False, weights="imagenet",
                           input_shape=input_shape, pooling="max")
        preprocess_fn = preprocess_densenet
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    base.trainable = False
    return base, preprocess_fn

# ─── CLASSIFIERS DEFINITION ───────────────────────────────────────────────────
def get_classical_classifiers(seed, classes_count):
    clfs = {
        "SVM":           SVC(kernel="rbf", probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "AdaBoost":      AdaBoostClassifier(n_estimators=100, random_state=seed),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "Bagging":       BaggingClassifier(n_estimators=100, random_state=seed),
        "ANN":           MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)
    }
    if XGBOOST_AVAILABLE:
        clfs["XGBoost"] = XGBClassifier(n_estimators=100, random_state=seed,
                                       use_label_encoder=False, eval_metric="logloss")
    else:
        # Log skipped warning once on first seed instantiation
        if seed == 42:
            print(f"\n[WARNING] Skipping XGBoost classifier because it failed to load. Reason: {XGBOOST_ERROR}\n")
    return clfs

# ─── EVALUATION HELPER ────────────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred):
    return {
        "acc":       round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, average="weighted") * 100, 2),
        "kappa":     round(cohen_kappa_score(y_true, y_pred) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average="weighted") * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted") * 100, 2)
    }

# ─── PLOTTING ─────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, model_name, seed, classes, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"{model_name} Confusion Matrix (Seed {seed})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}_seed{seed}.png")
    plt.savefig(path, dpi=150)
    plt.close()

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Feature Extraction CT Classifier Pipeline")
    parser.add_argument("--backbone", type=str, default="xception", choices=["xception", "densenet121"],
                        help="Feature extractor backbone model")
    parser.add_argument("--split-mode", type=str, default="stratified", choices=["split_dir", "stratified"],
                        help="Data loading and splitting mode. 'stratified' (default) uses the "
                             "deduplicated corpus with the same 80/10/10 protocol as the Track A "
                             "fine-tuning scripts. 'split_dir' reproduces the original leaky setup "
                             "(train/ vs test/ directories, no deduplication) and is kept only for "
                             "comparison.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs for training neural network classifiers (LSTM, Bi-LSTM)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for feature extraction and training")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21, 99, 123],
                        help="List of random seeds to evaluate over")
    parser.add_argument("--output-dir", type=str, default="./feature_extraction_outputs",
                        help="Directory to save generated outputs and figures")
    args = parser.parse_args()

    # ─── CONFIGURATION ────────────────────────────────────────────────────────
    CLASSES = ["normal", "pancreatic_tumor"]
    SEEDS = args.seeds
    
    # Setup image sizes natively matching the backbones
    IMG_SIZE = (299, 299) if args.backbone.lower() == "xception" else (224, 224)
    
    output_subdir = os.path.join(args.output_dir, f"{args.backbone.lower()}_{args.split_mode}")
    os.makedirs(output_subdir, exist_ok=True)

    print("=" * 75)
    print(f"  CT Feature Extraction Pipeline | Backbone: {args.backbone.upper()}")
    print(f"  Split Mode: {args.split_mode.upper()} | Output: {output_subdir}")
    print("=" * 75)

    # ─── LOAD DATASETS ────────────────────────────────────────────────────────
    if args.split_mode == "split_dir":
        print("[INFO] Loading datasets from 'train' and 'test' folders separately...")
        train_paths, train_labels = get_image_paths_labels("./train", CLASSES)
        test_paths, test_labels = get_image_paths_labels("./test", CLASSES)
        print(f"  Found {len(train_paths)} train images and {len(test_paths)} test images.")
    else:
        print("[INFO] Loading deduplicated corpus for stratified seed splitting...")
        all_paths, all_labels = du.load_merged_dataset(dedupe=True)
        print(f"  Found {len(all_paths)} unique images after content-hash deduplication.")

    # Build the feature extractor base model
    extractor, preprocess_fn = build_feature_extractor(args.backbone, IMG_SIZE)

    # Dictionary to collect results across all seeds
    # Structure: { model_name: [ { metric_dict }, ... ] }
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'-'*65}")
        print(f"  [STARTING SEED {seed}]")
        print(f"{'-'*65}")
        set_seed(seed)

        # ─── TRAIN / VAL / TEST SPLITTING ─────────────────────────────────────
        if args.split_mode == "split_dir":
            # Train on all ./train images
            tr_p, tr_l = train_paths, train_labels
            # Split ./test images 50% validation, 50% test
            va_p, te_p, va_l, te_l = train_test_split(
                test_paths, test_labels, test_size=0.5, random_state=seed, stratify=test_labels
            )
        else:
            # 80/10/10 stratified split, identical to the Track A fine-tuning scripts
            tr_p, va_p, te_p, tr_l, va_l, te_l = du.stratified_split(
                all_paths, all_labels, seed
            )

        print(f"  Splits size -> Train: {len(tr_p)} | Val: {len(va_p)} | Test: {len(te_p)}")

        # ─── IMAGE LOADING & PREPROCESSING ────────────────────────────────────
        print("  [Step 1] Loading and preprocessing images...")
        X_tr_raw = load_and_preprocess_images(tr_p, IMG_SIZE, preprocess_fn)
        X_va_raw = load_and_preprocess_images(va_p, IMG_SIZE, preprocess_fn)
        X_te_raw = load_and_preprocess_images(te_p, IMG_SIZE, preprocess_fn)

        # ─── FEATURE EXTRACTION ───────────────────────────────────────────────
        print("  [Step 2] Extracting high-dimensional deep features...")
        X_tr = extractor.predict(X_tr_raw, batch_size=args.batch_size, verbose=1)
        X_va = extractor.predict(X_va_raw, batch_size=args.batch_size, verbose=1)
        X_te = extractor.predict(X_te_raw, batch_size=args.batch_size, verbose=1)

        print(f"  Extracted Features shape -> Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

        # Free some raw images memory
        del X_tr_raw, X_va_raw, X_te_raw

        # ─── CLASSIFICATION — CLASSICAL + ANN ─────────────────────────────────
        print("  [Step 3] Training Classical and Neural classifiers...")
        classifiers = get_classical_classifiers(seed, len(CLASSES))

        for clf_name, clf in classifiers.items():
            print(f"    Training {clf_name}...")
            clf.fit(X_tr, tr_l)

            # Predict
            p_tr = clf.predict(X_tr)
            p_va = clf.predict(X_va)
            p_te = clf.predict(X_te)

            # Metrics
            train_m = calculate_metrics(tr_l, p_tr)
            val_m   = calculate_metrics(va_l, p_va)
            test_m  = calculate_metrics(te_l, p_te)

            metrics_summary = {
                "seed": seed,
                "train_acc": train_m["acc"],
                "val_acc": val_m["acc"],
                "test_acc": test_m["acc"],
                "f1": test_m["f1"],
                "kappa": test_m["kappa"],
                "recall": test_m["recall"],
                "precision": test_m["precision"]
            }
            all_results.setdefault(clf_name, []).append(metrics_summary)
            
            # Save confusion matrix for classical models on seed 42 (primary seed)
            if seed == 42:
                cm = confusion_matrix(te_l, p_te)
                save_confusion_matrix(cm, clf_name, seed, CLASSES, output_subdir)

        # ─── CLASSIFICATION — RECURRENT MODELS (LSTM & BI-LSTM) ────────────────
        # Reshape extracted features to 3D: (samples, time_steps=1, feature_dimension)
        X_tr_3d = X_tr.reshape(-1, 1, X_tr.shape[1])
        X_va_3d = X_va.reshape(-1, 1, X_va.shape[1])
        X_te_3d = X_te.reshape(-1, 1, X_te.shape[1])

        rnn_layers = [
            ("LSTM",    LSTM(64)),
            ("Bi-LSTM", Bidirectional(LSTM(64)))
        ]

        for rnn_name, rnn_layer in rnn_layers:
            print(f"    Training {rnn_name}...")
            model = Sequential([
                rnn_layer,
                Dense(32, activation="relu"),
                Dense(len(CLASSES), activation="softmax")
            ])
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            # Train the recurrent model
            history = model.fit(
                X_tr_3d, tr_l,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(X_va_3d, va_l),
                shuffle=False,
                verbose=0
            )

            # Predict probabilities and convert to label predictions
            p_tr = np.argmax(model.predict(X_tr_3d, verbose=0), axis=1)
            p_va = np.argmax(model.predict(X_va_3d, verbose=0), axis=1)
            p_te = np.argmax(model.predict(X_te_3d, verbose=0), axis=1)

            # Metrics
            train_m = calculate_metrics(tr_l, p_tr)
            val_m   = calculate_metrics(va_l, p_va)
            test_m  = calculate_metrics(te_l, p_te)

            metrics_summary = {
                "seed": seed,
                "train_acc": train_m["acc"],
                "val_acc": val_m["acc"],
                "test_acc": test_m["acc"],
                "f1": test_m["f1"],
                "kappa": test_m["kappa"],
                "recall": test_m["recall"],
                "precision": test_m["precision"]
            }
            all_results.setdefault(rnn_name, []).append(metrics_summary)

            # Save curves & confusion matrix for seed 42
            if seed == 42:
                cm = confusion_matrix(te_l, p_te)
                save_confusion_matrix(cm, rnn_name, seed, CLASSES, output_subdir)
                
                # Plot training history curves
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history.history["accuracy"], label="Train")
                ax1.plot(history.history["val_accuracy"], label="Val")
                ax1.set_title(f"{rnn_name} Accuracy (Seed {seed})")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Accuracy")
                ax1.legend()

                ax2.plot(history.history["loss"], label="Train")
                ax2.plot(history.history["val_loss"], label="Val")
                ax2.set_title(f"{rnn_name} Loss (Seed {seed})")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Loss")
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_subdir, f"curves_{rnn_name.lower().replace('-', '')}_seed{seed}.png"), dpi=150)
                plt.close()

    # ─── AGGREGATE & DISPLAY RESULTS ──────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"            FINAL COMPARATIVE RESULTS ({args.backbone.upper()})")
    print("=" * 85)
    
    summary_rows = []
    
    # Table header
    header = f"{'Model':<15} {'Train Acc':>11} {'Val Acc':>11} {'Test Acc':>11} {'F1 Score':>11} {'Kappa':>9} {'Recall':>9} {'Precision':>10}"
    print(header)
    print("-" * len(header))

    detailed_rows = []
    metrics_keys = ["train_acc", "val_acc", "test_acc", "f1", "kappa", "recall", "precision"]

    for model_name, seeds_runs in all_results.items():
        # Store detailed row metrics per seed for export
        for run in seeds_runs:
            detailed_rows.append({
                "Model": model_name,
                "Seed": run["seed"],
                **{k: run[k] for k in metrics_keys}
            })

        # Calculate statistics (Mean ± Std Dev) across seeds
        stats = {}
        for key in metrics_keys:
            vals = [run[key] for run in seeds_runs]
            stats[f"{key}_mean"] = np.mean(vals)
            stats[f"{key}_std"]  = np.std(vals)

        # Print Mean ± SD format
        print(f"{model_name:<15} "
              f"{stats['train_acc_mean']:>4.1f}±{stats['train_acc_std']:>3.1f}% "
              f"{stats['val_acc_mean']:>4.1f}±{stats['val_acc_std']:>3.1f}% "
              f"{stats['test_acc_mean']:>4.1f}±{stats['test_acc_std']:>3.1f}% "
              f"{stats['f1_mean']:>4.1f}±{stats['f1_std']:>3.1f}% "
              f"{stats['kappa_mean']:>4.1f}±{stats['kappa_std']:>3.1f}% "
              f"{stats['recall_mean']:>4.1f}±{stats['recall_std']:>3.1f}% "
              f"{stats['precision_mean']:>4.1f}±{stats['precision_std']:>3.1f}%")

        summary_rows.append({
            "Model": model_name,
            **{f"{k}_mean": round(stats[f"{k}_mean"], 2) for k in metrics_keys},
            **{f"{k}_std": round(stats[f"{k}_std"], 2) for k in metrics_keys}
        })

    print("=" * len(header))

    # Save outputs to CSV files
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(os.path.join(output_subdir, "detailed_seed_results.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_subdir, "summary_results.csv"), index=False)
    
    print(f"\n[SUCCESS] Pipeline runs completed successfully!")
    print(f"  Summary saved to -> {output_subdir}/summary_results.csv")
    print(f"  Seed runs saved to -> {output_subdir}/detailed_seed_results.csv")
    print(f"  Confusion matrices and curves saved to directory.")


if __name__ == "__main__":
    main()
