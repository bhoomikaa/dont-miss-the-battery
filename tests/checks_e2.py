perfect = {}
for ds in (val_ds, test_ds):
    for iid in ds.ids:
        b, l = ds.gt(iid)
        perfect[iid] = {"boxes": b, "scores": np.full(len(b), 0.99, np.float32), "labels": l}
wrong = {i: {**p, "labels": (p["labels"] % K) + 1} for i, p in perfect.items()}
for name, P in [("PERFECT", perfect), ("WRONG-LABEL", wrong)]:
    rep = safety_report(P, P)
    m = coco_metrics({i: P[i] for i in test_ds.ids}, test_gt)
    print(name, "mAP50", round(m["mAP50"], 3), "| any-miss", rep["class_agnostic"]["test_at_thr"]["miss_rate"],
          "| type-miss", rep["class_aware"]["test_at_thr"]["miss_rate"], "| zero-event bound", rep["class_agnostic"].get("miss_upper95_zero_events"),
          "| CI", rep["class_aware"]["miss_ci"])
