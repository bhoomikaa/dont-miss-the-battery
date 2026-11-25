"""CPU smoke test for E1_yolo_rtdetr.py on a tiny subset."""
import os, sys, re, shutil, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
work = os.path.join(HERE, "run_E1"); os.makedirs(work, exist_ok=True); os.chdir(work)
src = open(sys.argv[1], encoding="utf-8").read()
def sub(old, new):
    global src
    assert old in src, old
    src = src.replace(old, new)
root = sys.argv[2].replace("\\", "/")
open(os.path.join(root, ".done"), "w").close()
sub('ROOT = "/content/recybat24"', f'ROOT = "{root}"')
sub("!pip -q install ultralytics==8.4.216", "pass")
src = re.sub(r"^EPOCHS = \d+", "EPOCHS = 1", src, flags=re.M)
src = re.sub(r"^EPOCHS_DETR = \d+", "EPOCHS_DETR = 1", src, flags=re.M)
src = re.sub(r"^N_BOOT = \d+", "N_BOOT = 50", src, flags=re.M)
sub("workers=2", "workers=0")
sub('test_ds = BatteryDet("val", test_ids)\n',
    'test_ds = BatteryDet("val", test_ids)\n'
    'train_ds.ids = train_ds.ids[::180][:8]; val_ds.ids = val_ds.ids[::60][:12]; test_ds.ids = test_ds.ids[::60][:12]\n')
# put the real E2 results where the notebook expects them, to test the merge
e2 = os.path.join(work, "outputs", "E2_safety"); os.makedirs(e2, exist_ok=True)
for f in os.listdir(sys.argv[3]):
    if f.endswith(".json"): shutil.copy(os.path.join(sys.argv[3], f), e2)
if len(sys.argv) > 4: src += chr(10) + open(sys.argv[4], encoding="utf-8").read()
exec(compile(src, "E1_yolo_rtdetr.py", "exec"), {"__name__": "__main__"})
print("HARNESS: completed without errors")
