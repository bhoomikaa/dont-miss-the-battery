# Round-trip: YOLO label files -> predictions -> our COCO + safety metrics must be perfect.
def from_labels(ds, sub):
    out = {}
    for iid in ds.ids:
        W = ds.imgs[iid].get("width", IMG_SIZE); H = ds.imgs[iid].get("height", IMG_SIZE)
        rows = np.loadtxt(os.path.splitext(yolo_name[(sub, iid)].replace("/images/", "/labels/"))[0] + ".txt", ndmin=2)
        c, cx, cy, w, h = rows.T if len(rows) else [np.zeros(0)] * 5
        b = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1).astype(np.float32)
        out[iid] = {"boxes": b, "scores": np.ones(len(b), np.float32), "labels": c.astype(np.int64) + 1}
    return out
pv, pt = from_labels(val_ds, "val"), from_labels(test_ds, "test")
m = coco_metrics(pt, test_gt); s = safety_report(pv, pt)
print("ROUNDTRIP mAP50", m["mAP50"], "mAP50_95", round(m["mAP50_95"], 4), "type miss", s["class_aware"]["test_at_thr"]["miss_rate"])
assert m["mAP50"] == 1.0 and m["mAP50_95"] > 0.99 and s["class_aware"]["test_at_thr"]["miss_rate"] == 0
# label convention check against the trained model's names
assert list(YOLO(os.path.join(run_dir("yolo11s"), "weights", "best.pt")).names.values()) == CLASSES[1:]
print("CHECKS: passed")
