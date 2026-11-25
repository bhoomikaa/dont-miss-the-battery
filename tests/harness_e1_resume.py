"""Resume test: YOLO only, 2 epochs; with INTERRUPT=1 the run dies right after epoch 1 is saved."""
import os, sys, re, shutil, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
work = os.path.join(HERE, "run_E1_resume"); os.makedirs(work, exist_ok=True); os.chdir(work)
src = open(sys.argv[1], encoding="utf-8").read()
def sub(old, new):
    global src
    assert old in src, old
    src = src.replace(old, new)
root = sys.argv[2].replace("\\", "/")
open(os.path.join(root, ".done"), "w").close()
sub('ROOT = "/content/recybat24"', f'ROOT = "{root}"')
sub("!pip -q install ultralytics==8.4.216", "pass")
src = re.sub(r"^EPOCHS = \d+", "EPOCHS = 2", src, flags=re.M)
src = re.sub(r"^N_BOOT = \d+", "N_BOOT = 50", src, flags=re.M)
sub('    "rtdetr_l": {"weights": "rtdetr-l.pt", "batch": 8,  "epochs": "EPOCHS_DETR"},\n', "")
sub("workers=2", "workers=0")
sub('test_ds = BatteryDet("val", test_ids)\n',
    'test_ds = BatteryDet("val", test_ids)\n'
    'train_ds.ids = train_ds.ids[::180][:8]; val_ds.ids = val_ds.ids[::60][:12]; test_ds.ids = test_ds.ids[::60][:12]\n')
if os.environ.get("INTERRUPT"):
    sub('        m = Model(cfg["weights"])\n',
        '        m = Model(cfg["weights"])\n'
        '        def _die(tr):\n'
        '            if tr.epoch == 0: raise SystemExit("SIMULATED DISCONNECT")\n'
        '        m.add_callback("on_model_save", _die)\n')
exec(compile(src, "E1_yolo_rtdetr.py", "exec"), {"__name__": "__main__"})
print("HARNESS: completed without errors")
