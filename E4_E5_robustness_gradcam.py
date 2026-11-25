# %% [markdown]
# # Don't Miss the Battery: E4 robustness to image corruptions and E5 Grad-CAM
# **Group 3 (Bhavyasree Kondi, Bhoomika Lnu, Sanjana Reddy Katham, Sharan Somshekhar Patil), DATA-255**
#
# **E4.** Sorting-line cameras see blur, poor light, compression and dirty or partly covered batteries. The four
# trained detectors (Faster R-CNN, Faster R-CNN + repeat-factor sampling, YOLO11-s, RT-DETR-l) are run on the
# test set under **8 corruptions x 5 severities**. Each detector keeps the operating threshold it got on the
# **clean validation** set (precision >= 0.90), the way a deployed system would. For every condition we report
# the missed-battery rate (any type), the type-aware miss rate (missed or wrong chemistry type), precision and
# mAP, with 95% intervals that resample whole batteries. The **unsafe severity** of a corruption is the first
# level where more than 1% of batteries go unnoticed.
#
# **E5.** Grad-CAM on Faster R-CNN shows where the model looks when it gets a prismatic battery right versus when
# it calls it a pouch, and on batteries lost under corruption. EigenCAM gives the matching view for RT-DETR.
# A simple number backs the pictures: the share of the heat map that falls inside the battery's box.
#
# **Needs:** the E2 and E1 notebooks finished (weights are loaded from Drive).
# **Before running:** Runtime > Change runtime type > T4 GPU, then Runtime > Run all (~1.5 to 2 h).
# Every finished condition is saved to Drive, so after a disconnect, Run all again and it continues.

# %%
!pip -q install ultralytics==8.4.216

# %%
# ---- Config ----
SEED = 42
IMG_SIZE = 640
IOU_MATCH = 0.5
TARGET_PRECISION = 0.90
MISS_COST = 10
COST_RATIOS = list(range(1, 21))
N_BOOT = 1000
CORRUPTIONS = ["gaussian_blur", "motion_blur", "gaussian_noise", "low_light", "low_contrast", "jpeg", "occlusion", "dirt"]
SEVERITIES = [1, 2, 3, 4, 5]
SEV = {                                    # parameter per severity 1..5
    "gaussian_blur":  [1, 2, 3, 5, 7],                 # Gaussian sigma (px)
    "motion_blur":    [5, 9, 15, 23, 31],              # blur length (px), random direction
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],  # noise std (fraction of 255)
    "low_light":      [0.6, 0.45, 0.3, 0.2, 0.1],      # brightness factor
    "low_contrast":   [0.6, 0.45, 0.3, 0.2, 0.1],      # contrast factor
    "jpeg":           [30, 18, 10, 6, 3],              # JPEG quality
    "occlusion":      [0.1, 0.2, 0.3, 0.4, 0.5],       # share of each battery's box covered
    "dirt":           [10, 25, 50, 90, 150],           # number of dirt spots
}
MISS_LIMIT = 0.01        # unsafe if more than 1% of batteries go unnoticed
TYPE_MISS_LIMIT = 0.05   # secondary: unsafe for sorting if more than 5% are missed or mistyped
MAX_CAM_GT = 400         # cap on prismatic batteries analysed with Grad-CAM
DATA_URL = "https://zenodo.org/api/records/15490523/files/recybat24.tar.gz/content"
ROOT = "/content/recybat24"

import os, json, random, time, glob, re, tarfile, math, contextlib, io, urllib.request, shutil
from collections import defaultdict, Counter
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_ARG = 0 if dev.type == "cuda" else "cpu"
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| device", dev)
if dev.type == "cuda":
    print(torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime > Change runtime type > T4 GPU")

# %%
# ---- Folders: weights from E2 and E1 on Drive, outputs to E4_E5_robustness ----
try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/dont_miss_the_battery"
except ImportError:
    BASE = "./outputs"
E2_DIR = os.path.join(BASE, "E2_safety")
E1_DIR = os.path.join(BASE, "E1_detectors")
OUT = os.path.abspath(os.path.join(BASE, "E4_E5_robustness"))
WEIGHTS = {
    "frcnn":     os.path.join(E2_DIR, "baseline_best.pt"),
    "frcnn_rfs": os.path.join(E2_DIR, "repeat_factor_best.pt"),
    "yolo11s":   os.path.join(E1_DIR, "yolo11s", "weights", "best.pt"),
    "rtdetr_l":  os.path.join(E1_DIR, "rtdetr_l", "weights", "best.pt"),
}
missing = [p for p in WEIGHTS.values() if not os.path.exists(p)] + [p for p in [os.path.join(E2_DIR, "split.json")] if not os.path.exists(p)]
assert not missing, f"Run the E2 and E1 notebooks first. Missing: {missing}"
os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
COND_DIR = os.path.join(OUT, "conditions"); os.makedirs(COND_DIR, exist_ok=True)
print("Outputs ->", OUT)

# %%
# ---- Download + extract RecyBat24 (Zenodo record 15490523, CC BY-NC-ND 4.0) ----
if not os.path.exists(os.path.join(ROOT, ".done")):
    os.makedirs(ROOT, exist_ok=True)
    tgz = "/content/recybat24.tar.gz"
    if not os.path.exists(tgz):
        print("Downloading ~263 MB ...")
        urllib.request.urlretrieve(DATA_URL, tgz)
    with tarfile.open(tgz) as t:
        try:
            t.extractall(ROOT, filter="data")
        except TypeError:
            t.extractall(ROOT)
    open(os.path.join(ROOT, ".done"), "w").close()

# ---- Detection annotations (the instance-segmentation file repeats every battery 3x, so skip it) ----
coco_files = {}
for p in glob.glob(ROOT + "/**/*.json", recursive=True):
    low = p.lower()
    if "segmentation" in os.path.basename(low):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, dict) and {"images", "annotations", "categories"} <= d.keys():
        if "train" in low:
            coco_files["train"] = p
        elif "val" in low:
            coco_files["val"] = p
assert {"train", "val"} <= coco_files.keys(), coco_files
coco = {k: json.load(open(v)) for k, v in coco_files.items()}
for k, d in coco.items():
    print(k, "| images:", len(d["images"]), "| boxes:", len(d["annotations"]))

img_index = defaultdict(list)
for p in glob.glob(ROOT + "/**/*", recursive=True):
    if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img_index[os.path.basename(p)].append(p)

def resolve(file_name, split):
    direct = os.path.join(os.path.dirname(coco_files[split]), file_name)
    if os.path.exists(direct):
        return direct
    cands = img_index.get(os.path.basename(file_name), [])
    for c in cands:
        if split in c.lower():
            return c
    if cands:
        return cands[0]
    raise FileNotFoundError(file_name)

used = sorted({a["category_id"] for s in coco.values() for a in s["annotations"]})
cat_names = {c["id"]: c["name"] for c in coco["train"]["categories"]}
cat2label = {cid: i + 1 for i, cid in enumerate(used)}            # 1..K (0 = background)
CLASSES = ["__background__"] + [cat_names.get(cid, str(cid)) for cid in used]
K = len(CLASSES) - 1
print("Classes:", CLASSES[1:])

# %%
# ---- Leak-free split (identical to the Faster R-CNN notebooks): released val -> val + test by battery ----
GROUP_REGEX = r"^(\d+)_\d+_(?:[A-Za-z]_)?\d+_([A-Za-z]+)"

def battery_id(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(GROUP_REGEX, stem)
    return "_".join(m.groups()) if m else stem

from sklearn.model_selection import StratifiedGroupKFold
val_imgs = coco["val"]["images"]
groups = [battery_id(i["file_name"]) for i in val_imgs]
types = [g.rsplit("_", 1)[-1] for g in groups]
vi, ti = next(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED).split(val_imgs, types, groups=groups))
val_ids = [val_imgs[i]["id"] for i in vi]
test_ids = [val_imgs[i]["id"] for i in ti]
split_e2 = os.path.join(E2_DIR, "split.json")
s = json.load(open(split_e2)); val_ids, test_ids = s["val_ids"], s["test_ids"]   # exact images of the E1/E2 runs
print("Loaded the E2 (Faster R-CNN) split from Drive")
bat_of = {i["id"]: battery_id(i["file_name"]) for i in val_imgs}
vb, tb = {bat_of[i] for i in val_ids}, {bat_of[i] for i in test_ids}
assert not (vb & tb) and len(val_ids) + len(test_ids) == len(val_imgs)
print(f"val {len(val_ids)} imgs / {len(vb)} batteries | test {len(test_ids)} imgs / {len(tb)} batteries")
json.dump({"val_ids": val_ids, "test_ids": test_ids, "seed": SEED}, open(os.path.join(OUT, "split.json"), "w"))

# %%
# ---- Ground truth access (same class as the Faster R-CNN notebooks, used for evaluation) ----
class BatteryDet:
    def __init__(self, split, img_ids=None):
        d = coco[split]
        self.split = split
        self.imgs = {i["id"]: i for i in d["images"]}
        self.ids = list(img_ids) if img_ids is not None else list(self.imgs)
        self.anns = defaultdict(list)
        for a in d["annotations"]:
            self.anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.ids)

    def gt(self, iid):
        boxes, labels = [], []
        for a in self.anns[iid]:
            x, y, w, h = a["bbox"]
            if w >= 1 and h >= 1:
                boxes.append([x, y, x + w, y + h]); labels.append(cat2label[a["category_id"]])
        return np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64)

    def battery(self, iid):
        return battery_id(self.imgs[iid]["file_name"])

    def path(self, iid):
        return resolve(self.imgs[iid]["file_name"], self.split)

train_ds = BatteryDet("train")
val_ds = BatteryDet("val", val_ids)
test_ds = BatteryDet("val", test_ids)
print(len(train_ds), len(val_ds), len(test_ds))

# %%
# ---- COCO metrics (identical to the Faster R-CNN notebooks) ----
from torchvision.ops import box_iou, nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def make_gt(ds):
    images, anns = [], []
    for iid in ds.ids:
        info = ds.imgs[iid]
        images.append({"id": iid, "width": info.get("width", IMG_SIZE), "height": info.get("height", IMG_SIZE)})
        b, l = ds.gt(iid)
        for bb, ll in zip(b, l):
            w, h = float(bb[2] - bb[0]), float(bb[3] - bb[1])
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": int(ll),
                         "bbox": [float(bb[0]), float(bb[1]), w, h], "area": w * h, "iscrowd": 0})
    c = COCO()
    c.dataset = {"images": images, "annotations": anns,
                 "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES) if i > 0]}
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c

def coco_metrics(preds, gt):
    res = [{"image_id": iid, "category_id": int(l), "score": float(s),
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
           for iid, p in preds.items() for b, s, l in zip(p["boxes"], p["scores"], p["labels"])]
    if not res:
        return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(res), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]
    ap = lambda p: float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    per_class = {CLASSES[k + 1]: {"AP50_95": ap(prec[:, :, k, 0, -1]), "AP50": ap(prec[0, :, k, 0, -1])} for k in range(K)}
    return {"mAP50_95": float(E.stats[0]), "mAP50": float(E.stats[1]), "mAP75": float(E.stats[2]), "per_class": per_class}

# %%
# ---- Safety evaluation (identical to the Faster R-CNN notebooks) ----
THS = np.round(np.arange(0.01, 1.0, 0.01), 2)

def match(preds, ds, class_aware):
    G = {"score": [], "cls": [], "pred": [], "bat": []}
    FA = {"score": [], "label": [], "bat": []}
    for iid in ds.ids:
        gb, gl = ds.gt(iid); bat = ds.battery(iid)
        p = preds[iid]; pb, ps, pl = torch.from_numpy(p["boxes"]), torch.from_numpy(p["scores"]), p["labels"]
        if not class_aware and len(pb):
            keep = nms(pb, ps, 0.5); pb, ps, pl = pb[keep], ps[keep], pl[keep.numpy()]
        ious = box_iou(pb, torch.from_numpy(gb)).numpy() if len(gb) and len(pb) else np.zeros((len(pb), len(gb)))
        g_score = np.full(len(gb), -1.0); g_pred = np.zeros(len(gb), int)
        for j in torch.argsort(ps, descending=True).numpy():
            cand = ious[j].copy(); cand[g_score >= 0] = -1
            if class_aware:
                cand[gl != pl[j]] = -1
            k = int(cand.argmax()) if len(gb) else -1
            if k >= 0 and cand[k] >= IOU_MATCH:
                g_score[k] = float(ps[j]); g_pred[k] = int(pl[j])
            else:
                FA["score"].append(float(ps[j])); FA["label"].append(int(pl[j])); FA["bat"].append(bat)
        G["score"] += g_score.tolist(); G["cls"] += gl.tolist(); G["pred"] += g_pred.tolist(); G["bat"] += [bat] * len(gb)
    return {k: np.array(v) for k, v in G.items()}, {k: np.array(v) for k, v in FA.items()}

def counts(G, FA, t):
    tp = int((G["score"] >= t).sum()); fp = int((FA["score"] >= t).sum()); n = len(G["score"])
    fn = n - tp
    prec = tp / max(tp + fp, 1); rec = tp / max(n, 1)
    return {"thr": float(t), "tp": tp, "fp": fp, "fn": fn, "n": n, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "miss_rate": fn / max(n, 1)}

def pick_threshold_precision(G, FA):
    rows = [counts(G, FA, t) for t in THS]
    ok = [r for r in rows if r["precision"] >= TARGET_PRECISION]
    best = min(ok, key=lambda r: (r["miss_rate"], -r["precision"])) if ok else max(rows, key=lambda r: r["precision"])
    return best["thr"]

def pick_threshold_cost(G, FA, c):
    return min(THS, key=lambda t: (lambda r: c * r["fn"] + r["fp"])(counts(G, FA, t)))

def bootstrap_miss(G, FA, t, seed=SEED):
    bats = np.unique(G["bat"])
    per = {b: (int(((G["bat"] == b) & (G["score"] >= t)).sum()), int((G["bat"] == b).sum()),
               int(((FA["bat"] == b) & (FA["score"] >= t)).sum())) for b in bats}
    rng = np.random.default_rng(seed); miss, prec = [], []
    for _ in range(N_BOOT):
        s = rng.choice(bats, len(bats), replace=True)
        tp = sum(per[b][0] for b in s); n = sum(per[b][1] for b in s); fp = sum(per[b][2] for b in s)
        miss.append(1 - tp / max(n, 1)); prec.append(tp / max(tp + fp, 1))
    out = {"miss_ci": [float(np.percentile(miss, 2.5)), float(np.percentile(miss, 97.5))],
           "precision_ci": [float(np.percentile(prec, 2.5)), float(np.percentile(prec, 97.5))],
           "n_batteries": int(len(bats))}
    if counts(G, FA, t)["fn"] == 0:
        out["miss_upper95_zero_events"] = float(1 - 0.05 ** (1 / len(bats)))
    return out

def ece(G, FA, min_score=0.05, bins=10):
    s = np.concatenate([G["score"][G["score"] >= min_score], FA["score"][FA["score"] >= min_score]])
    y = np.concatenate([np.ones((G["score"] >= min_score).sum()), np.zeros((FA["score"] >= min_score).sum())])
    edges = np.linspace(min_score, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (s >= lo) & ((s < hi) | (hi == 1))
        if b.any():
            e += b.mean() * abs(y[b].mean() - s[b].mean())
    return float(e)

def confusion(G, FA, t):
    cm = np.zeros((K + 1, K + 1), int)
    for sc, c, p in zip(G["score"], G["cls"], G["pred"]):
        cm[c - 1, p - 1 if sc >= t else K] += 1
    for sc, l in zip(FA["score"], FA["label"]):
        if sc >= t:
            cm[K, l - 1] += 1
    return cm

def safety_report(val_preds, test_preds):
    rep = {}
    for mode in ["class_agnostic", "class_aware"]:
        Gv, Fv = match(val_preds, val_ds, mode == "class_aware")
        Gt, Ft = match(test_preds, test_ds, mode == "class_aware")
        t90 = pick_threshold_precision(Gv, Fv)
        r = {"thr_from_val": t90, "test_at_thr": counts(Gt, Ft, t90), **bootstrap_miss(Gt, Ft, t90),
             "cost": {}, "ece": ece(Gt, Ft)}
        for c in COST_RATIOS:
            tc = pick_threshold_cost(Gv, Fv, c); ct = counts(Gt, Ft, tc)
            r["cost"][c] = {"thr_from_val": float(tc), "cwe_per_battery": (c * ct["fn"] + ct["fp"]) / max(ct["n"], 1),
                            "miss_rate": ct["miss_rate"], "precision": ct["precision"], "fp": ct["fp"], "fn": ct["fn"]}
        r["pr_curve_test"] = [counts(Gt, Ft, t) for t in THS]
        if mode == "class_agnostic":
            cm = confusion(Gt, Ft, t90); r["confusion"] = cm.tolist()
            r["per_class_miss"] = {CLASSES[c]: float(((Gt["cls"] == c) & (Gt["score"] < t90)).sum() / max((Gt["cls"] == c).sum(), 1))
                                   for c in range(1, K + 1)}
            pi, po = CLASSES.index("prismatic") if "prismatic" in CLASSES else None, CLASSES.index("pouch") if "pouch" in CLASSES else None
            if pi and po:
                r["prismatic_as_pouch"] = float(cm[pi - 1, po - 1] / max(cm[pi - 1].sum(), 1))
                r["pouch_as_prismatic"] = float(cm[po - 1, pi - 1] / max(cm[po - 1].sum(), 1))
        rep[mode] = r
    return rep

val_gt, test_gt = make_gt(val_ds), make_gt(test_ds)

# %%
# ---- Load the four trained detectors ----
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO, RTDETR

def build_frcnn():   # identical to the E2 notebook
    m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=IMG_SIZE, max_size=IMG_SIZE)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, K + 1)
    m.roi_heads.score_thresh = 0.001
    m.roi_heads.detections_per_img = 100
    return m

LOADED = {}
for name in ["frcnn", "frcnn_rfs"]:
    m = build_frcnn(); m.load_state_dict(torch.load(WEIGHTS[name], map_location="cpu")); LOADED[name] = ("frcnn", m.to(dev).eval())
LOADED["yolo11s"] = ("ultra", YOLO(WEIGHTS["yolo11s"]))
LOADED["rtdetr_l"] = ("ultra", RTDETR(WEIGHTS["rtdetr_l"]))
NAMES = list(LOADED)
assert list(LOADED["yolo11s"][1].names.values()) == CLASSES[1:], "class order differs from training"

@torch.no_grad()
def predict_arrays(name, imgs, ids):
    """imgs: list of BGR uint8 arrays. Same settings as E1/E2: conf 0.001, at most 100 boxes, labels 1..K."""
    kind, m = LOADED[name]; preds = {}
    bs = 8 if kind == "frcnn" else 16
    for i in range(0, len(ids), bs):
        chunk = imgs[i:i + bs]
        if kind == "frcnn":
            x = [torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float().div(255).to(dev) for im in chunk]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = m(x)
            out = [{"boxes": o["boxes"].float().cpu().numpy(), "scores": o["scores"].float().cpu().numpy(),
                    "labels": o["labels"].cpu().numpy().astype(np.int64)} for o in out]
        else:
            res = m.predict(list(chunk), imgsz=IMG_SIZE, conf=0.001, iou=0.7, max_det=100, verbose=False, device=DEVICE_ARG)
            out = [{"boxes": r.boxes.xyxy.cpu().numpy().astype(np.float32), "scores": r.boxes.conf.cpu().numpy().astype(np.float32),
                    "labels": r.boxes.cls.cpu().numpy().astype(np.int64) + 1} for r in res]
        for iid, o in zip(ids[i:i + bs], out):
            preds[iid] = o
    return preds

test_imgs = [cv2.imread(test_ds.path(iid)) for iid in test_ds.ids]
print("test images in memory:", len(test_imgs), test_imgs[0].shape)

# %%
# ---- Operating thresholds from the CLEAN validation set (precision >= 0.90), fixed for every corruption ----
thr_path = os.path.join(OUT, "thresholds.json")
if os.path.exists(thr_path):
    THR = json.load(open(thr_path)); print("Loaded thresholds")
else:
    THR = {}
    for name in NAMES:
        vp = {}
        for i in range(0, len(val_ds.ids), 128):   # stream validation images to keep memory low
            ids = val_ds.ids[i:i + 128]
            vp.update(predict_arrays(name, [cv2.imread(val_ds.path(iid)) for iid in ids], ids))
        THR[name] = {mode: float(pick_threshold_precision(*match(vp, val_ds, mode == "class_aware")))
                     for mode in ["class_agnostic", "class_aware"]}
    json.dump(THR, open(thr_path, "w"), indent=1)
print(THR)

# %%
# ---- Corruptions (deterministic: the same image + corruption + severity always gives the same picture) ----
def corrupt(img, kind, sev, iid, boxes):
    v = SEV[kind][sev - 1]
    rng = np.random.default_rng([SEED, int(iid), CORRUPTIONS.index(kind), sev])
    f = img.astype(np.float32); H, W = img.shape[:2]
    if kind == "gaussian_blur":
        out = cv2.GaussianBlur(img, (0, 0), v)
    elif kind == "motion_blur":
        k = np.zeros((v, v), np.float32); k[v // 2, :] = 1
        M = cv2.getRotationMatrix2D((v / 2 - 0.5, v / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        k = cv2.warpAffine(k, M, (v, v)); out = cv2.filter2D(img, -1, k / k.sum())
    elif kind == "gaussian_noise":
        out = f + rng.normal(0, v * 255, img.shape)
    elif kind == "low_light":
        out = f * v
    elif kind == "low_contrast":
        mu = f.mean((0, 1), keepdims=True); out = mu + (f - mu) * v
    elif kind == "jpeg":
        out = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(v)])[1], cv2.IMREAD_COLOR)
    elif kind == "occlusion":            # a flat patch covering a share v of every battery's box
        out = img.copy()
        for x1, y1, x2, y2 in boxes:
            w, h = (x2 - x1) * math.sqrt(v), (y2 - y1) * math.sqrt(v)
            ox, oy = rng.uniform(x1, x2 - w), rng.uniform(y1, y2 - h)
            out[int(oy):int(math.ceil(oy + h)), int(ox):int(math.ceil(ox + w))] = int(rng.integers(40, 200))
    elif kind == "dirt":                 # brown/black spots, soft edges
        mask = np.zeros((H, W), np.float32); col = np.zeros_like(f)
        for _ in range(v):
            c = (int(rng.integers(15, 60)), int(rng.integers(30, 80)), int(rng.integers(45, 110)))   # BGR, brownish
            ctr = (int(rng.uniform(0, W)), int(rng.uniform(0, H))); ax = (int(rng.uniform(3, 20)), int(rng.uniform(3, 20)))
            ang = float(rng.uniform(0, 180))
            cv2.ellipse(mask, ctr, ax, ang, 0, 360, 1.0, -1); cv2.ellipse(col, ctr, ax, ang, 0, 360, c, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None] * 0.9; col = cv2.GaussianBlur(col, (0, 0), 1.5)
        out = f * (1 - mask) + col * mask
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8)

# Example sheet: one test image (with a prismatic battery if possible) under every corruption and severity
ex_i = next((i for i, iid in enumerate(test_ds.ids) if CLASSES.index("prismatic") in test_ds.gt(iid)[1]), 0)
ex_id, ex_img = test_ds.ids[ex_i], test_imgs[ex_i]
fig, axes = plt.subplots(len(CORRUPTIONS), len(SEVERITIES) + 1, figsize=(2.2 * (len(SEVERITIES) + 1), 2.2 * len(CORRUPTIONS)))
for r, c in enumerate(CORRUPTIONS):
    for col, s in enumerate([0] + SEVERITIES):
        im = ex_img if s == 0 else corrupt(ex_img, c, s, ex_id, test_ds.gt(ex_id)[0])
        a = axes[r, col]
        a.imshow(im[..., ::-1]); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title("clean" if s == 0 else f"severity {s}", fontsize=9)
        if s == 0: a.set_ylabel(c.replace("_", " "), fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "E4_corruption_examples.jpg"), dpi=80); plt.show()

# %%
# ---- E4: run every detector on every condition (resumable) ----
def evaluate(preds, name):
    out = {"coco": coco_metrics(preds, test_gt)}
    for mode, key in [("class_agnostic", "any"), ("class_aware", "type")]:
        G, FA = match(preds, test_ds, mode == "class_aware")
        t = THR[name][mode]
        out[key] = {**counts(G, FA, t), **bootstrap_miss(G, FA, t)}
        if mode == "class_agnostic":
            cm = confusion(G, FA, t); pi, po = CLASSES.index("prismatic"), CLASSES.index("pouch")
            out["prismatic_as_pouch"] = float(cm[pi - 1, po - 1] / max(cm[pi - 1].sum(), 1))
    return out

CONDS = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
ROB = {}
t_all = time.time()
for ci, (c, s) in enumerate(CONDS):
    path = os.path.join(COND_DIR, f"{c}_{s}.json")
    if os.path.exists(path):
        ROB[(c, s)] = json.load(open(path)); continue
    t0 = time.time()
    imgs = test_imgs if c == "clean" else [corrupt(im, c, s, iid, test_ds.gt(iid)[0]) for iid, im in zip(test_ds.ids, test_imgs)]
    ROB[(c, s)] = {name: evaluate(predict_arrays(name, imgs, test_ds.ids), name) for name in NAMES}
    json.dump(ROB[(c, s)], open(path, "w"), default=float)
    del imgs
    print(f"[{ci + 1}/{len(CONDS)}] {c:14s} s{s} | " +
          " | ".join(f"{n} any {ROB[(c, s)][n]['any']['miss_rate']:.3f} type {ROB[(c, s)][n]['type']['miss_rate']:.3f}" for n in NAMES) +
          f" | {time.time() - t0:.0f}s (total {(time.time() - t_all) / 60:.1f} min)")

# Sanity check: the clean condition should reproduce the E1/E2 test numbers (tiny JPEG-decoder differences are possible)
ref = {}
for p in [os.path.join(E1_DIR, "E1_summary.json")]:
    if os.path.exists(p):
        ref = {r["model"]: r for r in json.load(open(p))}
for n in NAMES:
    if n in ref:
        print(f"{n:10s} clean type-miss {ROB[('clean', 0)][n]['type']['miss_rate']:.3f} (E1/E2 table: {ref[n]['type_miss@P90']:.3f}) | "
              f"mAP50 {ROB[('clean', 0)][n]['coco']['mAP50']:.3f} ({ref[n]['mAP50']:.3f})")

# %%
# ---- E4 summary: unsafe severity, mean error under corruption, relative corruption error ----
def first_unsafe(name, c, key, limit):
    for s in SEVERITIES:
        if ROB[(c, s)][name][key]["miss_rate"] > limit:
            return s
    return None

summary = {}
for n in NAMES:
    corr = [ROB[(c, s)][n] for c in CORRUPTIONS for s in SEVERITIES]
    summary[n] = {
        "clean_any_miss": ROB[("clean", 0)][n]["any"]["miss_rate"], "clean_type_miss": ROB[("clean", 0)][n]["type"]["miss_rate"],
        "clean_mAP50": ROB[("clean", 0)][n]["coco"]["mAP50"],
        "mean_any_miss_corrupted": float(np.mean([r["any"]["miss_rate"] for r in corr])),
        "mean_type_miss_corrupted": float(np.mean([r["type"]["miss_rate"] for r in corr])),
        "mean_precision_corrupted": float(np.mean([r["type"]["precision"] for r in corr])),
        "mean_mAP50_corrupted": float(np.mean([r["coco"]["mAP50"] for r in corr])),
        "unsafe_severity_any": {c: first_unsafe(n, c, "any", MISS_LIMIT) for c in CORRUPTIONS},
        "unsafe_severity_type": {c: first_unsafe(n, c, "type", TYPE_MISS_LIMIT) for c in CORRUPTIONS},
        "type_miss_by_corruption": {c: [ROB[(c, s)][n]["type"]["miss_rate"] for s in SEVERITIES] for c in CORRUPTIONS},
        "any_miss_by_corruption": {c: [ROB[(c, s)][n]["any"]["miss_rate"] for s in SEVERITIES] for c in CORRUPTIONS},
    }
# Relative corruption error (in the spirit of ImageNet-C's mCE): summed type-aware misses / Faster R-CNN's
base = sum(ROB[(c, s)]["frcnn"]["type"]["fn"] for c in CORRUPTIONS for s in SEVERITIES)
for n in NAMES:
    summary[n]["relative_corruption_error_vs_frcnn"] = sum(ROB[(c, s)][n]["type"]["fn"] for c in CORRUPTIONS for s in SEVERITIES) / max(base, 1)
json.dump(summary, open(os.path.join(OUT, "E4_summary.json"), "w"), indent=1, default=float)

hdr = f"{'model':10s} {'clean any':>9s} {'clean type':>10s} {'corr any':>9s} {'corr type':>10s} {'corr mAP50':>10s} {'rel. CE':>8s}"
print(hdr); print("-" * len(hdr))
for n in NAMES:
    s_ = summary[n]
    print(f"{n:10s} {s_['clean_any_miss']:9.3f} {s_['clean_type_miss']:10.3f} {s_['mean_any_miss_corrupted']:9.3f} "
          f"{s_['mean_type_miss_corrupted']:10.3f} {s_['mean_mAP50_corrupted']:10.3f} {s_['relative_corruption_error_vs_frcnn']:8.2f}")
print("\nUnsafe severity (first level with > 1% of batteries unnoticed; '-' = safe up to severity 5)")
print(f"{'model':10s} " + " ".join(f"{c[:12]:>12s}" for c in CORRUPTIONS))
for n in NAMES:
    print(f"{n:10s} " + " ".join(f"{(summary[n]['unsafe_severity_any'][c] or '-'):>12}" for c in CORRUPTIONS))

# %%
# ---- E4 figures ----
colors = dict(zip(NAMES, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]))
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for a, c in zip(axes.ravel(), CORRUPTIONS):
    xs = [0] + SEVERITIES
    for n in NAMES:
        a.plot(xs, [ROB[("clean", 0)][n]["any"]["miss_rate"]] + [ROB[(c, s)][n]["any"]["miss_rate"] for s in SEVERITIES],
               "o-", c=colors[n], ms=3, label=f"{n} any")
        a.plot(xs, [ROB[("clean", 0)][n]["type"]["miss_rate"]] + [ROB[(c, s)][n]["type"]["miss_rate"] for s in SEVERITIES],
               "s--", c=colors[n], ms=3, alpha=0.6, label=f"{n} type")
    a.axhline(MISS_LIMIT, c="gray", lw=0.8, ls=":"); a.set_title(c.replace("_", " ")); a.set_xticks(xs)
    a.set_xlabel("severity (0 = clean)")
axes[0, 0].set_ylabel("miss rate"); axes[1, 0].set_ylabel("miss rate")
axes[0, 0].legend(fontsize=7, ncol=2)
plt.suptitle("Missed batteries under corruption (solid: any type, dashed: missed or mistyped; thresholds fixed from clean val)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "E4_miss_vs_severity.png"), dpi=120); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 3.8))
for a, key, title in [(ax[0], "unsafe_severity_any", "Unsafe severity: > 1% of batteries unnoticed"),
                      (ax[1], "unsafe_severity_type", "Unsafe severity: > 5% missed or mistyped")]:
    M = np.array([[summary[n][key][c] or 6 for c in CORRUPTIONS] for n in NAMES], float)
    a.imshow(M, cmap="RdYlGn", vmin=1, vmax=6)
    for i in range(len(NAMES)):
        for j in range(len(CORRUPTIONS)):
            a.text(j, i, "safe" if M[i, j] == 6 else int(M[i, j]), ha="center", va="center", fontsize=9)
    a.set_xticks(range(len(CORRUPTIONS))); a.set_xticklabels([c.replace("_", "\n") for c in CORRUPTIONS], fontsize=8)
    a.set_yticks(range(len(NAMES))); a.set_yticklabels(NAMES); a.set_title(title, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "E4_unsafe_severity.png"), dpi=120); plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(NAMES))
ax.bar(x - 0.2, [summary[n]["clean_type_miss"] for n in NAMES], 0.4, label="clean")
ax.bar(x + 0.2, [summary[n]["mean_type_miss_corrupted"] for n in NAMES], 0.4, label="mean over 40 corrupted conditions")
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("type-aware miss rate"); ax.legend()
ax.set_title("Clean vs corrupted type-aware miss rate (test)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "E4_clean_vs_corrupted.png"), dpi=120); plt.show()

# %%
# ---- E5: Grad-CAM (Faster R-CNN, box-specific) and EigenCAM (Faster R-CNN and RT-DETR, class-agnostic) ----
def to_tensor(img_bgr):
    return torch.from_numpy(np.ascontiguousarray(img_bgr[..., ::-1])).permute(2, 0, 1).float().div(255).to(dev)

def norm_cam(cam):
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()

def eigencam(A):
    """First principal component of the activations (Muhammad & Yeasin, 2020), sign fixed to agree with the mean activation."""
    C, H, W = A.shape
    X = A.reshape(C, -1).T.float(); X = X - X.mean(0)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    p = X @ Vh[0]
    if torch.corrcoef(torch.stack([p, A.reshape(C, -1).float().mean(0)]))[0, 1] < 0:
        p = -p
    return norm_cam(torch.relu(p).reshape(H, W))

class FRCNNCam:
    """Grad-CAM for one detection: gradient of that detection's class score w.r.t. the last ResNet stage (stride 32)."""
    def __init__(self, model):
        self.m, self.acts, self.grads = model, None, None
        model.backbone.body.layer4.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))
    def forward(self, img_bgr):
        self.m.eval()
        with torch.enable_grad():
            return self.m([to_tensor(img_bgr)])[0]
    def gradcam(self, score):
        self.m.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        A, G = self.acts[0], self.grads[0]
        return norm_cam(torch.relu((G.mean((1, 2))[:, None, None] * A).sum(0)))
    def eigen(self):
        return eigencam(self.acts[0].detach())

class UltraEigenCam:
    """EigenCAM on the last stride-32 feature map before the detection head of an Ultralytics model."""
    def __init__(self, ultra_model):
        self.net = ultra_model.model.to(dev).float().eval()
        layers = list(self.net.model)
        outs = {}
        hs = [l.register_forward_hook(lambda mod, i, o, k=k: outs.__setitem__(k, o)) for k, l in enumerate(layers)]
        with torch.no_grad():
            self.net(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=dev))
        for h in hs:
            h.remove()
        cands = [k for k, o in outs.items() if torch.is_tensor(o) and o.dim() == 4 and o.shape[-1] == IMG_SIZE // 32 and k < len(layers) - 1]
        self.k = cands[-1]; self.acts = None
        layers[self.k].register_forward_hook(lambda mod, i, o: setattr(self, "acts", o))
        print(f"EigenCAM layer {self.k}: {type(layers[self.k]).__name__} {tuple(outs[self.k].shape)}")
    def __call__(self, img_bgr):
        with torch.no_grad():
            self.net(to_tensor(img_bgr)[None])
        return eigencam(self.acts[0])

def energy_in_box(cam, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    return float(cam[max(y1, 0):y2, max(x1, 0):x2].sum() / max(cam.sum(), 1e-8))

def best_overlap(det, gt_box):
    """Highest-scoring detection that overlaps the battery (IoU >= 0.5), any class, any score. Index or None."""
    if len(det["boxes"]) == 0:
        return None
    iou = box_iou(torch.as_tensor(det["boxes"]).float(), torch.as_tensor(gt_box[None]).float())[:, 0].cpu().numpy()
    ok = np.where(iou >= IOU_MATCH)[0]
    return int(ok[np.argmax(det["scores"][ok])]) if len(ok) else None

fr = FRCNNCam(LOADED["frcnn"][1])
rt = UltraEigenCam(LOADED["rtdetr_l"][1])
t_fr = THR["frcnn"]["class_agnostic"]
PRISM, POUCH = CLASSES.index("prismatic"), CLASSES.index("pouch")

cam_rows, examples = [], {"confused": [], "correct": []}
for idx, iid in enumerate(test_ds.ids):
    gb, gl = test_ds.gt(iid)
    if PRISM not in gl:
        continue
    out = fr.forward(test_imgs[idx])
    det = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
    fr_eigen = fr.eigen()
    rt_cam = rt(test_imgs[idx])
    for b, l in zip(gb, gl):
        if l != PRISM or len(cam_rows) >= MAX_CAM_GT:
            continue
        j = best_overlap(det, b)
        if j is None or det["scores"][j] < t_fr:
            outcome = "missed"
        else:
            outcome = "correct" if det["labels"][j] == l else ("as_pouch" if det["labels"][j] == POUCH else "other_type")
        row = {"image_id": int(iid), "battery": test_ds.battery(iid), "outcome": outcome,
               "eigencam_frcnn_in_box": energy_in_box(fr_eigen, b), "eigencam_rtdetr_in_box": energy_in_box(rt_cam, b)}
        if j is not None:
            g = fr.gradcam(out["scores"][j])
            row.update({"gradcam_frcnn_in_box": energy_in_box(g, b), "score": float(det["scores"][j]), "pred": CLASSES[int(det["labels"][j])]})
            key = "confused" if outcome == "as_pouch" else ("correct" if outcome == "correct" else None)
            if key:   # keep only the 3 highest-scoring examples per group (heat maps are large)
                examples[key].append((float(det["scores"][j]), idx, b, det["boxes"][j], CLASSES[int(det["labels"][j])], g, rt_cam))
                examples[key] = sorted(examples[key], key=lambda e: -e[0])[:3]
        cam_rows.append(row)
    if len(cam_rows) >= MAX_CAM_GT:
        break

from scipy.stats import mannwhitneyu
groups_ = {o: [r for r in cam_rows if r["outcome"] == o] for o in ["correct", "as_pouch", "other_type", "missed"]}
cam_summary = {"n": {o: len(v) for o, v in groups_.items()}, "threshold": t_fr}
for key in ["gradcam_frcnn_in_box", "eigencam_frcnn_in_box", "eigencam_rtdetr_in_box"]:
    cam_summary[key] = {o: {"mean": float(np.mean([r[key] for r in v if key in r])) if any(key in r for r in v) else None,
                            "median": float(np.median([r[key] for r in v if key in r])) if any(key in r for r in v) else None}
                        for o, v in groups_.items()}
    a_ = [r[key] for r in groups_["correct"] if key in r]; b_ = [r[key] for r in groups_["as_pouch"] if key in r]
    cam_summary[key]["p_value_correct_vs_as_pouch"] = float(mannwhitneyu(a_, b_).pvalue) if len(a_) > 1 and len(b_) > 1 else None
json.dump({"summary": cam_summary, "rows": cam_rows}, open(os.path.join(OUT, "E5_cam.json"), "w"), indent=1, default=float)
print(json.dumps(cam_summary, indent=1, default=float))

# %%
# ---- E5 cases under corruption: batteries Faster R-CNN loses at severity 3 of its worst corruption ----
worst = max(CORRUPTIONS, key=lambda c: ROB[(c, 3)]["frcnn"]["any"]["miss_rate"] if (c, 3) in ROB else -1)
examples["corrupted"] = []
for idx, iid in enumerate(test_ds.ids):
    if len(examples["corrupted"]) >= 2:
        break
    gb, gl = test_ds.gt(iid)
    if not len(gb):
        continue
    im = corrupt(test_imgs[idx], worst, 3, iid, gb)
    out = fr.forward(im); det = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
    for b, l in zip(gb, gl):
        j = best_overlap(det, b)
        if j is None or det["scores"][j] < t_fr:      # lost under corruption
            g = fr.gradcam(out["scores"][j]) if j is not None else np.zeros((IMG_SIZE, IMG_SIZE))
            examples["corrupted"].append((float(det["scores"][j]) if j is not None else 0.0, im, b,
                                          det["boxes"][j] if j is not None else None,
                                          CLASSES[int(det["labels"][j])] if j is not None else "none", g, rt(im)))
            break
print("worst corruption at severity 3 for Faster R-CNN:", worst, "| lost examples found:", len(examples["corrupted"]))

# %%
# ---- E5 figures ----
def overlay(img_bgr, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)[..., ::-1]

def draw(a, img_rgb, gt_box, det_box, title):
    a.imshow(img_rgb); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=8)
    for bx, colr in [(gt_box, "lime"), (det_box, "red")]:
        if bx is not None:
            a.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=colr, lw=1.5))

rt_names = LOADED["rtdetr_l"][1].names
cases = []
for sc, idx, b, db, lab, g, rc in sorted(examples["confused"], key=lambda e: -e[0])[:3]:
    cases.append(("prismatic called " + lab, test_imgs[idx], b, db, lab, sc, g, rc))
for sc, idx, b, db, lab, g, rc in sorted(examples["correct"], key=lambda e: -e[0])[:1]:
    cases.append(("prismatic, correct", test_imgs[idx], b, db, lab, sc, g, rc))
for sc, im, b, db, lab, g, rc in examples["corrupted"]:
    cases.append((f"lost under {worst.replace('_', ' ')} s3", im, b, db, lab, sc, g, rc))
if cases:
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.3 * len(cases)), squeeze=False)
    for r, (title, im, b, db, lab, sc, g, rc) in enumerate(cases):
        rp = predict_arrays("rtdetr_l", [im], [-1])[-1]; jr = best_overlap(rp, b)
        rt_lab = f"{CLASSES[int(rp['labels'][jr])]} {rp['scores'][jr]:.2f}" if jr is not None else "none"
        draw(axes[r, 0], im[..., ::-1], b, db, f"{title}\nFaster R-CNN: {lab} {sc:.2f} (green = truth)")
        draw(axes[r, 1], overlay(im, g), b, None, "Faster R-CNN Grad-CAM (that box's score)")
        draw(axes[r, 2], overlay(im, rc), b, rp["boxes"][jr] if jr is not None else None, f"RT-DETR EigenCAM | RT-DETR: {rt_lab}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "E5_cam_cases.jpg"), dpi=90); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, key, title in zip(ax, ["gradcam_frcnn_in_box", "eigencam_frcnn_in_box", "eigencam_rtdetr_in_box"],
                         ["Faster R-CNN Grad-CAM", "Faster R-CNN EigenCAM", "RT-DETR EigenCAM"]):
    data = [[r[key] for r in groups_[o] if key in r] for o in ["correct", "as_pouch", "missed"]]
    keep = [(d, o) for d, o in zip(data, ["correct", "called pouch", "missed"]) if len(d)]
    if keep:
        a.boxplot([d for d, _ in keep]); a.set_xticks(range(1, len(keep) + 1)); a.set_xticklabels([f"{o}\n(n={len(d)})" for d, o in keep])
    a.set_ylabel("share of heat inside the battery box"); a.set_title(f"{title}: prismatic batteries (Faster R-CNN outcome)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "E5_cam_energy.png"), dpi=120); plt.show()

# %%
# ---- Zip results + figures and download ----
pack = "/content/E4_E5_results" if os.path.isdir("/content") else "./E4_E5_results"
os.makedirs(pack, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".json"):
        shutil.copy(os.path.join(OUT, f), pack)
shutil.copytree(COND_DIR, os.path.join(pack, "conditions"), dirs_exist_ok=True)
shutil.copytree(FIG, os.path.join(pack, "figures"), dirs_exist_ok=True)
zip_path = shutil.make_archive(pack, "zip", pack)
print("Results zip:", zip_path)
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    pass
