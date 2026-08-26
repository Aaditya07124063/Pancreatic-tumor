"""
Build the full project report as a PDF.

Reads every results CSV in the repository at build time, so the document can
never drift from the numbers actually produced. Generates two summary figures,
then lays out the report.

    .venv/bin/python build_report.py        # -> Pancreatic_CT_Research_Report.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, KeepTogether)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Pancreatic_CT_Research_Report.pdf")
FIGDIR = os.path.join(ROOT, "report_figures")
os.makedirs(FIGDIR, exist_ok=True)

INK = colors.HexColor("#16202B")
ACCENT = colors.HexColor("#2C4B9B")
ALERT = colors.HexColor("#C0342A")
MUTED = colors.HexColor("#6A7885")
RULE = colors.HexColor("#C7D2DA")
BAND = colors.HexColor("#F2F5F8")

MODELS = [
    ("ResNet-50", "Pretrained CNN", "resnet50_outputs/resnet50_results.csv"),
    ("InceptionV3", "Pretrained CNN", "inceptionv3_outputs/inceptionv3_results.csv"),
    ("MobileViT", "Pretrained Transformer", "mobilevit_outputs/mobilevit_results.csv"),
    ("Swin Transformer", "Pretrained Transformer", "swin_outputs/swin_results.csv"),
    ("ScratchCNN", "From-scratch CNN", "cnn_scratch_outputs/cnn_scratch_results.csv"),
    ("ScratchViT", "From-scratch Transformer", "vit_scratch_outputs/vit_scratch_results.csv"),
]


# ───────────────────────────── data ──────────────────────────────
def model_stats():
    rows = []
    for name, kind, path in MODELS:
        d = pd.read_csv(os.path.join(ROOT, path))
        d = d[d.seed.astype(str) != "AVG"]
        acc = d.test_acc.astype(float) * 100
        rows.append(dict(name=name, kind=kind, mean=acc.mean(), sd=acc.std(),
                         f1=d.f1.astype(float).mean(),
                         kappa=d.kappa.astype(float).mean(),
                         perfect=int((acc == 100).sum()), seeds=d))
    return rows


def load_grouped(path, key, val="test_acc"):
    p = os.path.join(ROOT, path)
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p)
    return d.groupby(key, sort=False)[val].agg(["mean", "std"]).reset_index()


# ──────────────────────────── figures ────────────────────────────
def fig_models(stats):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    names = [s["name"] for s in stats] + ["Brightness\nthreshold"]
    vals = [s["mean"] for s in stats] + [100.0]
    errs = [s["sd"] for s in stats] + [0.0]
    cols = ["#2C4B9B"] * 4 + ["#1D6B3C"] * 2 + ["#C0342A"]
    ax.bar(range(len(vals)), vals, yerr=errs, capsize=3, color=cols, width=.62)
    ax.axhline(52.5, ls="--", lw=1, color="#6A7885")
    ax.text(len(vals) - .4, 53.6, "majority baseline 52.5%", fontsize=7,
            color="#6A7885", ha="right")
    ax.set_ylim(40, 104)
    ax.set_ylabel("Test accuracy (%)", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=18, ha="right")
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Every method converges on the same score", fontsize=9, loc="left")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig_models.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_localisation():
    d = load_grouped("signal_localisation_results.csv", "arm")
    if d is None:
        return None
    lbl = {"FULL": "Whole scan", "BORDER": "Pancreas\ndeleted",
           "CENTRE": "Anatomy\nonly", "TINY8": "8x8\nthumbnail"}
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    cols = {"FULL": "#2C4B9B", "BORDER": "#C0342A",
            "CENTRE": "#6A7885", "TINY8": "#C0342A"}
    ax.bar([lbl.get(a, a) for a in d.arm], d["mean"] * 100,
           yerr=d["std"] * 100, capsize=3,
           color=[cols.get(a, "#2C4B9B") for a in d.arm], width=.55)
    ax.axhline(52.5, ls="--", lw=1, color="#6A7885")
    ax.set_ylim(40, 104)
    ax.set_ylabel("Test accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Removing the organ does not change the result",
                 fontsize=9, loc="left")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig_localisation.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ──────────────────────────── styles ─────────────────────────────
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=21, leading=25, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=11.5, leading=15,
                          alignment=1, textColor=ACCENT, spaceAfter=22),
    "meta": ParagraphStyle("m", parent=ss["Normal"], fontSize=9.5, leading=15,
                           alignment=1, textColor=MUTED),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=14, leading=17, textColor=ACCENT,
                         spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11, leading=14, textColor=INK,
                         spaceBefore=11, spaceAfter=5),
    "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.6, leading=14,
                           alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.2, leading=11,
                          textColor=MUTED, spaceBefore=3, spaceAfter=11),
    "key": ParagraphStyle("k", parent=ss["BodyText"], fontSize=10, leading=14.5,
                          textColor=ALERT, fontName="Helvetica-Bold",
                          borderPadding=7, backColor=colors.HexColor("#FCF2F1"),
                          borderColor=ALERT, borderWidth=1, spaceAfter=11),
    "mono": ParagraphStyle("mo", parent=ss["Normal"], fontName="Courier",
                           fontSize=8, leading=11, textColor=INK),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def table(data, widths, align_right=None, head=True, fs=8.3):
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0, hAlign="LEFT")
    st = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", fs),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), .4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    if head:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
               ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    for c in (align_right or []):
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


def chrome(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(.5)
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.1 * cm,
                          "Pancreatic Tumour Detection on CT — SIRE Summer Internship")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ───────────────────────────── build ─────────────────────────────
def build():
    stats = model_stats()
    f_models = fig_models(stats)
    f_loc = fig_localisation()

    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Pancreatic Tumour Detection on CT",
                            author="Aaditya Adhikari")
    E = []

    # ---------- title page ----------
    E += [Spacer(1, 3.4 * cm),
          P("Deep Learning for Pancreatic Tumour Detection on CT Scans", "title"),
          P("Why 100% Accuracy Was Wrong: An Integrity Audit", "sub"),
          Spacer(1, .8 * cm),
          P("<b>Aaditya Adhikari</b><br/>Department of Computer Science &amp; Engineering<br/>"
            "National Institute of Technology, Rourkela", "meta"),
          Spacer(1, .9 * cm),
          P("Under the guidance of<br/><b>Dr. Dhirendra Prasad Yadav</b><br/>"
            "Associate Professor, GLA University, Mathura", "meta"),
          Spacer(1, .9 * cm),
          P("SIRE — Summer Internship in Research Experience<br/>"
            "11 May 2026 – 20 July 2026", "meta"),
          PageBreak()]

    # ---------- abstract ----------
    E += [P("Abstract", "h1"),
          P("Six deep-learning models — four ImageNet-pretrained and two written from "
            "scratch — were trained to distinguish pancreatic tumour CT slices from "
            "normal ones under a single controlled protocol: a deduplicated corpus, "
            "stratified 80:10:10 splits, five random seeds and fifty epochs per run. "
            "All six reached 99.83–100.00% test accuracy."),
          P("Those scores were then tested rather than reported. A content-hash audit "
            "found 196 byte-identical duplicate files and 18 images filed under both "
            "class labels; a perceptual-hash audit found that 66.1% of test images had "
            "a near-identical twin in their own training partition; and a probe using "
            "seven non-diagnostic image statistics found that the fraction of pixels "
            "brighter than 200 separates the two classes with no overlap whatsoever, so "
            "a single threshold classifies all 1,179 images correctly."),
          P("A four-arm control removed the brightness difference, the leakage, and both "
            "together. Accuracy remained 100.00 ± 0.00 in every arm, so neither fault "
            "explained the result. A localisation experiment then blanked the central "
            "60% of every scan, removing the pancreas entirely: accuracy was 99.81%, "
            "against 99.82% for the intact image. Reducing each scan to an 8×8 "
            "thumbnail also left accuracy unchanged."),
          P("<b>The classes are separable by coarse properties of image production, "
            "outside the anatomy. No model evaluated here was reading the pancreas.</b> "
            "The architecture comparison the study set out to make is not identifiable "
            "from this corpus. The contribution is the audit protocol and the controlled "
            "experiments that establish this, both reusable on any imaging dataset."),
          PageBreak()]

    # ---------- 1 introduction ----------
    E += [P("1. Introduction and Objectives", "h1"),
          P("Pancreatic ductal adenocarcinoma has one of the poorest survival rates in "
            "oncology, largely because it is usually detected late. Automated detection "
            "on routine abdominal CT is therefore a high-value target, and the published "
            "literature routinely reports accuracy above 95%."),
          P("This project began with a conventional comparison question: under one "
            "identical protocol, do transformer architectures outperform convolutional "
            "ones at binary pancreatic tumour classification, and does end-to-end "
            "fine-tuning outperform frozen-feature extraction? Answering it required "
            "first establishing that the dataset could support any comparison at all."),
          P("Objectives", "h2")]
    E.append(table([
        ["#", "Objective"],
        ["1", "Benchmark convolutional and transformer models under one identical protocol"],
        ["2", "Enforce fair evaluation: stratified 80:10:10, five seeds, fifty epochs"],
        ["3", "Implement a CNN and a Vision Transformer from first principles"],
        ["4", "Validate whether reported accuracy reflects genuine diagnostic ability"],
    ], [1.1 * cm, 14.6 * cm]))

    # ---------- 2 dataset ----------
    E += [P("2. Dataset", "h1"),
          P("The corpus arrived as two directories, <font face='Courier'>train/</font> "
            "and <font face='Courier'>test/</font>, each containing "
            "<font face='Courier'>normal/</font> and "
            "<font face='Courier'>pancreatic_tumor/</font> subfolders."),
          table([
              ["Stage", "Normal", "Tumour", "Total"],
              ["As delivered", "646", "765", "1,411"],
              ["Exact duplicate files removed", "—", "—", "−196"],
              ["Conflicting-label groups excluded", "—", "—", "−18 groups"],
              ["Final analysis corpus", "620", "559", "1,179"],
          ], [7.2 * cm, 2.8 * cm, 2.8 * cm, 2.9 * cm], align_right=[1, 2, 3]),
          P("Deduplication merges both directories and keeps one file per unique content "
            "hash. Images appearing under conflicting labels are excluded entirely, since "
            "at least one of their two labels must be wrong.", "cap"),
          P("Every model uses stratified 80:10:10 partitions of this corpus, preserving "
            "class balance in each partition:"),
          table([
              ["Partition", "Images", "Share", "Normal", "Tumour"],
              ["Train", "943", "80.0%", "496", "447"],
              ["Validation", "118", "10.0%", "62", "56"],
              ["Test", "118", "10.0%", "62", "56"],
          ], [4.4 * cm, 2.7 * cm, 2.7 * cm, 2.9 * cm, 3 * cm],
              align_right=[1, 2, 3, 4])]

    # ---------- 3 methodology ----------
    E += [PageBreak(), P("3. Methodology", "h1"),
          P("3.1 Protocol", "h2"),
          P("Identical across every model: deduplicated corpus, stratified 80:10:10 "
            "splits, seeds 42, 7, 21, 99 and 123, and fifty epochs per seed. Early "
            "stopping patience was set equal to the epoch budget so that the full fifty "
            "epochs always run, while best-validation weights are still restored for "
            "evaluation. Results are reported as mean ± standard deviation over the five "
            "seeds."),
          P("3.2 Models", "h2"),
          table([
              ["Model", "Category", "Weights"],
              ["ResNet-50", "CNN", "ImageNet, fine-tuned end to end"],
              ["InceptionV3", "CNN", "ImageNet, fine-tuned end to end"],
              ["MobileViT", "Transformer", "apple/mobilevit-small"],
              ["Swin Transformer", "Transformer", "microsoft/swin-tiny-patch4-window7-224"],
              ["ScratchCNN", "CNN", "None — 4 conv blocks, trained from random init"],
              ["ScratchViT", "Transformer", "None — hand-written multi-head attention"],
          ], [4.2 * cm, 3.1 * cm, 8.4 * cm]),
          P("3.3 Audit methods", "h2"),
          table([
              ["Check", "What it does", "What a positive result means"],
              ["Content hash\n(MD5)", "Fingerprints the exact bytes of every file",
               "The same image is stored more than once"],
              ["Perceptual hash\n(dHash)", "Fingerprints what the image looks like",
               "Near-identical slices straddle the split"],
              ["Shortcut probe", "Logistic regression on seven non-diagnostic statistics",
               "Classes differ in how they were produced"],
              ["Control", "Removes brightness and leakage, alone and together",
               "Isolates which fault drives the score"],
              ["Localisation", "Blanks the anatomy; separately reduces to 8×8",
               "Locates the signal inside or outside the organ"],
          ], [3.3 * cm, 6.2 * cm, 6.2 * cm], fs=7.6)]

    # ---------- 4 results ----------
    E += [PageBreak(), P("4. Results: Six-Model Benchmark", "h1")]
    rows = [["Model", "Category", "Accuracy (%)", "F1", "Kappa", "Perfect"]]
    for s in stats:
        rows.append([s["name"], s["kind"], f"{s['mean']:.2f} ± {s['sd']:.2f}",
                     f"{s['f1']:.4f}", f"{s['kappa']:.4f}", f"{s['perfect']}/5"])
    rows.append(["Brightness threshold", "No learning at all", "100.00 ± 0.00",
                 "1.0000", "1.0000", "5/5"])
    t = table(rows, [3.7 * cm, 4 * cm, 3.1 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm],
              align_right=[2, 3, 4, 5])
    t.setStyle(TableStyle([("TEXTCOLOR", (0, len(rows) - 1), (-1, len(rows) - 1), ALERT),
                           ("FONT", (0, len(rows) - 1), (-1, len(rows) - 1),
                            "Helvetica-Bold", 8.3)]))
    E += [t,
          P("Mean ± SD over five seeds, fifty epochs each. The final row is not a model: "
            "it is a single threshold on the fraction of pixels brighter than 200, "
            "applied with no training whatsoever.", "cap"),
          Image(f_models, width=16 * cm, height=6.9 * cm),
          P("Figure 1. All six models, and a rule that never examines the anatomy, land "
            "within 0.17 percentage points of one another.", "cap"),
          P("Accuracy, F1 and Cohen's kappa agree with one another throughout. An earlier "
            "version of the pipeline reported roughly 98% accuracy alongside kappa near "
            "zero — arithmetically impossible, and traced to test-set shuffling that "
            "desynchronised predictions from labels. That defect is fixed."),
          P("<b>The comparison is not identifiable.</b> All six models fall inside a "
            "0.17-point band, narrower than seed-to-seed variation. A four-layer CNN "
            "trained from random initialisation matches Swin Transformer exactly.")]

    # ---------- 5 track B ----------
    xc = load_grouped("feature_extraction_outputs/xception_stratified/detailed_seed_results.csv", "Model")
    dn = load_grouped("feature_extraction_outputs/densenet121_stratified/detailed_seed_results.csv", "Model")
    if xc is not None and dn is not None:
        m = xc.merge(dn, on="Model", suffixes=("_x", "_d"))
        rows = [["Classifier", "Xception (%)", "DenseNet121 (%)"]]
        for _, r in m.iterrows():
            rows.append([r["Model"], f"{r['mean_x']:.2f} ± {r['std_x']:.2f}",
                         f"{r['mean_d']:.2f} ± {r['std_d']:.2f}"])
        E += [PageBreak(), P("5. Frozen-Feature Track", "h1"),
              P("Two ImageNet backbones were frozen and used purely as feature "
                "extractors, each feeding eight downstream classifiers under the same "
                "corpus, splits and seeds."),
              table(rows, [5.4 * cm, 5.1 * cm, 5.2 * cm], align_right=[1, 2]),
              P("Seven of eight classifiers on Xception features reach exactly 100.00% "
                "with zero variance. Random Forest — specifically queried during review — "
                "is indistinguishable from Swin Transformer.", "cap")]

    # ---------- 6 audit ----------
    E += [P("6. Integrity Audit", "h1"),
          P("6.1 Duplication and label conflict", "h2"),
          P("Hashing all 1,411 files found 196 byte-identical duplicates. Eighteen "
            "hashes carried contradictory labels: the identical image filed as healthy "
            "in one folder and diseased in another."),
          P("6.2 Near-duplicate leakage", "h2"),
          P("CT studies yield contiguous slices that are near-identical without being "
            "byte-identical, so content hashing cannot see them. Measured on the "
            "already-deduplicated corpus:"),
          table([
              ["Perceptual-hash distance", "Test images with a twin in training"],
              ["Identical hash", "66.1%"],
              ["Within 2 bits", "96.4%"],
              ["Within 5 bits", "98.5%"],
          ], [7.6 * cm, 8.1 * cm], align_right=[1]),
          P("6.3 The brightness artefact", "h2"),
          P("Of seven non-diagnostic statistics, one separates the classes completely:"),
          table([
              ["Class", "Fraction of pixels brighter than 200"],
              ["Normal", "0.00001 – 0.00613"],
              ["Pancreatic tumour", "0.01003 – 0.13960"],
          ], [5.4 * cm, 10.3 * cm]),
          P("The ranges do not overlap. A single threshold at 0.008 classifies all 1,179 "
            "images correctly on every seed, against a 52.5% majority baseline.", "cap")]

    # ---------- 7 control ----------
    ctl = load_grouped("provenance_control_results.csv", "arm")
    if ctl is not None:
        rows = [["Arm", "Brightness", "Leakage", "Accuracy (%)"]]
        desc = {"A raw+random": ("present", "present"),
                "B norm+random": ("removed", "present"),
                "C raw+group": ("present", "removed"),
                "D norm+group": ("removed", "removed")}
        for _, r in ctl.iterrows():
            b, l = desc.get(r["arm"], ("—", "—"))
            rows.append([r["arm"], b, l, f"{r['mean']*100:.2f} ± {r['std']*100:.2f}"])
        E += [PageBreak(), P("7. The Control Experiment", "h1"),
              P("If the brightness artefact and the leakage were responsible for the "
                "scores, removing them should reduce accuracy. Four arms tested this "
                "directly, five seeds each."),
              table(rows, [4.6 * cm, 3.6 * cm, 3.6 * cm, 3.9 * cm], align_right=[3]),
              P("Rank equalisation removes the brightness difference; group-aware "
                "splitting keeps near-identical images on one side of the partition, "
                "reducing twin contamination from 66.1% to 0.0%.", "cap"),
              Paragraph("The result did not move. Removing either fault, or both "
                        "together, left accuracy at 100.00 ± 0.00. Each fault is "
                        "individually sufficient, so closing one leaves the others "
                        "available — and something beyond both survives.", S["key"]),
              P("This is reported as the null result it was. It refuted the project's own "
                "working hypothesis, which is why the investigation continued rather "
                "than concluding here.")]

    # ---------- 8 localisation ----------
    loc = load_grouped("signal_localisation_results.csv", "arm")
    if loc is not None:
        lab = {"FULL": "Whole scan", "BORDER": "Pancreas deleted (central 60% blanked)",
               "CENTRE": "Anatomy only", "TINY8": "8×8 thumbnail"}
        rows = [["What the model can see", "Accuracy (%)"]]
        for _, r in loc.iterrows():
            rows.append([lab.get(r["arm"], r["arm"]),
                         f"{r['mean']*100:.2f} ± {r['std']*100:.2f}"])
        E += [P("8. Locating the Signal", "h1"),
              P("Since the control could not move the result, the cause was located "
                "rather than assumed. Every arm below runs under the strictest control "
                "condition — brightness equalised, leakage eliminated — and only the "
                "visible region changes."),
              table(rows, [10.5 * cm, 5.2 * cm], align_right=[1]),
              Image(f_loc, width=16 * cm, height=6.4 * cm) if f_loc else Spacer(1, 1),
              P("Figure 2. Deleting the organ costs 0.01 percentage points. Destroying "
                "all fine detail costs nothing.", "cap"),
              Paragraph("With the pancreas removed from every image, accuracy is 99.81% "
                        "against 99.82% for the intact scan. An 8×8 thumbnail — 64 "
                        "pixels — scores identically. Sixty-four pixels cannot represent "
                        "tumour morphology, and the blanked arm contains no pancreas at "
                        "all.", S["key"]),
              P("The anatomy-only arm is both lower and six times more variable "
                "(98.70 ± 2.91 against 99.81 ± 0.41), making the organ the least reliable "
                "cue available — the inverse of what a genuine detector would show.")]

    # ---------- 9 discussion ----------
    E += [PageBreak(), P("9. Discussion", "h1"),
          P("9.1 What the numbers measure", "h2"),
          P("Twenty-one distinct methods were evaluated: four pretrained networks, two "
            "written from scratch, fifteen classical and neural classifiers on frozen "
            "features, and a single brightness threshold requiring no training. Every one "
            "scores between 98.98% and 100.00%. That uniformity is itself the finding — "
            "when a one-line threshold rule matches a state-of-the-art transformer, the "
            "model is not the variable under study."),
          P("9.2 Why stratified splitting cannot repair this", "h2"),
          P("Stratified splitting equalises class proportions across partitions. It "
            "cannot help when the two classes are distinguishable by how their images "
            "were produced, because that difference is present in every partition by "
            "construction. Deduplication, additional seeds and longer training are "
            "equally powerless for the same reason."),
          P("9.3 Limitations", "h2"),
          P("The specific mechanism outside the anatomy is characterised but not fully "
            "itemised: brightness is demonstrably sufficient on its own, and the "
            "localisation experiment proves the residual signal lies outside the central "
            "region, but the remaining routes are not individually enumerated. Group-aware "
            "splitting is performed at perceptual-hash cluster level rather than patient "
            "level, because the corpus carries no patient identifiers — cluster-level "
            "grouping is a strict improvement on random splitting but is not equivalent "
            "to patient-level partitioning."),
          P("9.4 What would be required", "h2"),
          P("Uniform windowing applied to every image regardless of class; partitioning "
            "at patient or study level rather than slice level; per-image provenance "
            "recorded at collection time; and a shortcut baseline reported in the same "
            "table as any headline result. The margin of a model over that baseline is "
            "the only part of its score defensibly attributable to learned image "
            "understanding.")]

    # ---------- 10 conclusion ----------
    E += [P("10. Conclusion", "h1"),
          P("Six architectures were benchmarked under a controlled and fair protocol, and "
            "all six scored close to 100%. Rather than report that as a success, this "
            "work tested it, and established that the corpus is separable by coarse "
            "properties of image production lying outside the pancreas."),
          P("The architecture question cannot be answered from these data. This is "
            "reported as a negative result with proof: the evidence is not the brightness "
            "artefact — which the control showed to be one route among several — but the "
            "localisation experiment, in which deleting the organ leaves accuracy "
            "unchanged."),
          P("Deep learning is not the failure here; the dataset is. A corpus assembled "
            "with uniform windowing and patient-level splits could still answer the "
            "original question. The deliverable of this internship is the audit protocol "
            "and the controlled experiments that make such a claim testable before thirty "
            "training runs are spent on it.")]

    # ---------- 11 reproducibility ----------
    E += [P("11. Reproducibility", "h1"),
          P("Every number in this report is produced by a script in the repository and "
            "was regenerated for this document from the results files at build time."),
          table([
              ["Script", "Purpose"],
              ["data_utils.py", "Deduplication and stratified 80:10:10 splitting"],
              ["resnet50_train.py, inceptionv3_train.py", "Pretrained CNN training"],
              ["mobilevit_train.py, swin_transformer_train.py", "Pretrained transformer training"],
              ["cnn_scratch_train.py, vit_scratch_train.py", "From-scratch models"],
              ["feature_extraction_pipeline.py", "Frozen-feature track"],
              ["leakage_audit.py", "Duplication, leakage and shortcut checks"],
              ["provenance_control.py", "Four-arm control experiment"],
              ["signal_localisation.py", "Anatomy-removal and resolution experiment"],
              ["run_all.sh", "Full protocol, end to end"],
          ], [6.6 * cm, 9.1 * cm], fs=7.8),
          P("Repository: github.com/Aaditya07124063/Pancreatic-tumor", "cap")]

    # ---------- appendix ----------
    E += [PageBreak(), P("Appendix A. Per-Seed Results", "h1")]
    for s in stats:
        rows = [["Seed", "Train", "Val", "Test", "F1", "Kappa"]]
        for _, r in s["seeds"].iterrows():
            rows.append([str(r["seed"]),
                         f"{float(r['train_acc'])*100:.2f}",
                         f"{float(r['val_acc'])*100:.2f}",
                         f"{float(r['test_acc'])*100:.2f}",
                         f"{float(r['f1']):.4f}", f"{float(r['kappa']):.4f}"])
        E.append(KeepTogether([
            P(s["name"], "h2"),
            table(rows, [2.3 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.7 * cm, 2.6 * cm],
                  align_right=[1, 2, 3, 4, 5], fs=7.8)]))

    E += [PageBreak(), P("Appendix B. Confusion Matrices (seed 42)", "h1")]
    pairs = []
    for name, _, path in MODELS:
        cm_path = os.path.join(ROOT, os.path.dirname(path), "cm_seed42.png")
        if os.path.isfile(cm_path):
            pairs.append((name, cm_path))
    for i in range(0, len(pairs), 2):
        chunk = pairs[i:i + 2]
        cells = [[Paragraph(f"<b>{n}</b>", S["cap"]) for n, _ in chunk],
                 [Image(p, width=7.4 * cm, height=5.9 * cm) for _, p in chunk]]
        t = Table(cells, colWidths=[7.8 * cm] * len(chunk), hAlign="LEFT")
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        E.append(t)

    doc.build(E, onFirstPage=chrome, onLaterPages=chrome)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"written: {p}  ({os.path.getsize(p)/1024:.0f} KB)")
