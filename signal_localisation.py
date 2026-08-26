"""
Where does the separability actually live?

The provenance-matched control failed to move the result: with the brightness
shortcut removed AND near-duplicate leakage removed, a 4-layer CNN still scores
100.00% +/- 0.0000 on every seed. So a third route through the data exists.

This asks one question: is that route inside the anatomy, or outside it?

Every arm runs under the STRICTEST condition from the control -- rank-equalised
intensities and group-aware splitting -- so brightness and leakage are already
eliminated in all of them. Only the visible region changes.

  FULL         whole image (reproduces control arm D)
  BORDER       central 60% blanked -- the pancreas is GONE, only the frame,
               background and any burned-in marks remain
  CENTRE       everything outside the central 60% blanked -- anatomy only
  TINY8        downsampled to 8x8 then back up -- all fine structure destroyed,
               only coarse layout survives

Reading it:
  BORDER near 100%  -> the signal is outside the anatomy. Provenance, decisively.
  TINY8  near 100%  -> no fine anatomical detail is needed. Provenance.
  BORDER at chance while CENTRE stays high -> the signal really is in the organ,
                       and the corpus may carry genuine pathological signal.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import data_utils as du
from cnn_scratch_train import (DEVICE, IMG_SIZE, BATCH_SIZE, LR,
                               PancreaticDataset, ScratchCNN, set_seed)
from provenance_control import (ProvenanceMatch, exact_dhash_groups,
                                group_stratified_split, evaluate)

FRAC = 0.60  # side length of the central box, as a fraction of the image


class Mask:
    """Blank the centre, or everything but the centre. Operates on a PIL image."""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        a = np.array(img)
        h, w = a.shape[:2]
        y0, y1 = int(h * (1 - FRAC) / 2), int(h * (1 + FRAC) / 2)
        x0, x1 = int(w * (1 - FRAC) / 2), int(w * (1 + FRAC) / 2)
        if self.mode == "border":          # remove the anatomy
            a[y0:y1, x0:x1] = 0
        elif self.mode == "centre":        # keep only the anatomy
            keep = a[y0:y1, x0:x1].copy()
            a[:] = 0
            a[y0:y1, x0:x1] = keep
        return Image.fromarray(a)


class Tiny:
    """Destroy fine structure: downsample hard, then scale back up."""

    def __init__(self, n=8):
        self.n = n

    def __call__(self, img):
        return img.resize((self.n, self.n), Image.BILINEAR).resize(
            (IMG_SIZE, IMG_SIZE), Image.NEAREST)


def make_transform(arm, augment):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    # Every arm is provenance-matched, exactly as in control arm D.
    steps = [ProvenanceMatch(), transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if arm == "BORDER":
        steps.append(Mask("border"))
    elif arm == "CENTRE":
        steps.append(Mask("centre"))
    elif arm == "TINY8":
        steps.append(Tiny(8))
    if augment:
        steps += [transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), norm])


def run_arm(arm, paths, labels, groups, epochs):
    print(f"\n{'='*72}\n  ARM {arm}\n{'='*72}", flush=True)
    rows = []
    for seed in du.SEEDS:
        set_seed(seed)
        tr, va, te = group_stratified_split(labels, groups, seed)
        mk = lambda idx, aug: DataLoader(
            PancreaticDataset(paths[idx], labels[idx], make_transform(arm, aug)),
            batch_size=BATCH_SIZE, shuffle=aug, num_workers=0)
        tr_ld, va_ld, te_ld = mk(tr, True), mk(va, False), mk(te, False)

        model = ScratchCNN(num_classes=len(du.CLASSES)).to(DEVICE)
        crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LR)
        best_va, best_state = -1.0, None
        for _ in range(epochs):
            model.train()
            for x, y in tr_ld:
                opt.zero_grad()
                crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
                opt.step()
            va, _, _ = evaluate(model, va_ld)
            if va > best_va:
                best_va, best_state = va, {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        acc, f1, kappa = evaluate(model, te_ld)
        rows.append(dict(arm=arm, seed=seed, n_test=len(te),
                         test_acc=acc, f1=f1, kappa=kappa))
        print(f"  seed {seed:>3}: n_test={len(te):>3} | test_acc={acc:.4f} "
              f"F1={f1:.4f} kappa={kappa:.4f}", flush=True)
    df = pd.DataFrame(rows)
    print(f"  MEAN test_acc={df.test_acc.mean():.4f} +/- {df.test_acc.std():.4f}",
          flush=True)
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=du.EPOCHS)
    args = ap.parse_args()

    paths, labels = du.load_merged_dataset(dedupe=True)
    groups = exact_dhash_groups(paths)
    print(f"Corpus: {len(paths)} images | {len(np.unique(groups))} dHash clusters")
    print(f"All arms: rank-equalised + group-split (control arm D conditions)\n")

    out = pd.concat([run_arm(a, paths, labels, groups, args.epochs)
                     for a in ["FULL", "BORDER", "CENTRE", "TINY8"]],
                    ignore_index=True)
    out.to_csv("signal_localisation_results.csv", index=False)

    print(f"\n{'='*72}\n  SUMMARY ({args.epochs} epochs, {len(du.SEEDS)} seeds, ScratchCNN)\n{'='*72}")
    s = out.groupby("arm").agg(test_acc=("test_acc", "mean"),
                               sd=("test_acc", "std"),
                               kappa=("kappa", "mean")).reindex(
        ["FULL", "BORDER", "CENTRE", "TINY8"]).reset_index()
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  BORDER contains no pancreas. If it scores near FULL, the corpus is")
    print("  separable by something outside the anatomy entirely.")


if __name__ == "__main__":
    main()
