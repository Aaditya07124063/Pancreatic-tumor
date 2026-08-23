"""Print average results across all trained models."""
import os
import pandas as pd

RESULT_FILES = [
    ("ResNet50 (pretrained)", "./resnet50_outputs/resnet50_results.csv"),
    ("InceptionV3 (pretrained)", "./inceptionv3_outputs/inceptionv3_results.csv"),
    ("MobileViT (pretrained)", "./mobilevit_outputs/mobilevit_results.csv"),
    ("Swin Transformer (pretrained)", "./swin_outputs/swin_results.csv"),
    ("CNN (scratch)", "./cnn_scratch_outputs/cnn_scratch_results.csv"),
    ("ViT (scratch)", "./vit_scratch_outputs/vit_scratch_results.csv"),
]

COLS = ["test_acc", "precision", "recall", "f1", "kappa", "val_acc", "train_acc"]


def main():
    print("=" * 100)
    print(f"{'Model':<30} {'Test Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Kappa':>10}")
    print("=" * 100)
    rows = []
    for name, path in RESULT_FILES:
        if not os.path.isfile(path):
            print(f"{name:<30} {'(not run yet)':>10}")
            continue
        df = pd.read_csv(path)
        avg = df[df["seed"].astype(str) == "AVG"]
        if avg.empty:
            avg = df.iloc[[-1]]
        r = avg.iloc[0]
        print(f"{name:<30} {r['test_acc']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['kappa']:>10.4f}")
        rows.append({"model": name, **{c: r[c] for c in COLS if c in r}})
    print("=" * 100)
    if rows:
        pd.DataFrame(rows).to_csv("./final_results_summary.csv", index=False)
        print("Saved ./final_results_summary.csv")


if __name__ == "__main__":
    main()
