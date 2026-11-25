"""CPU smoke test for E3_zero_shot.py: tiny subset, severities 1 and 5, stand-in weights."""
import os, sys, re, shutil, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = "C:/Users/bhoom/OneDrive/Desktop/dont-miss-the-battery"
work = os.path.join(HERE, "run_E3"); os.makedirs(work, exist_ok=True); os.chdir(work)
src = open(os.path.join(PROJ, "E3_zero_shot.py"), encoding="utf-8").read()
def sub(old, new):
    global src
    assert old in src, old
    src = src.replace(old, new)
root = "C:/Users/bhoom/datasets/recybat24/recybat24"
open(os.path.join(root, ".done"), "w").close()
sub('ROOT = "/content/recybat24"', f'ROOT = "{root}"')
sub("!pip -q install ultralytics==8.4.216 transformers==5.24.0", "pass")
src = re.sub(r"^N_BOOT = \d+", "N_BOOT = 50", src, flags=re.M)
sub('test_ds = BatteryDet("val", test_ids)\n',
    'test_ds = BatteryDet("val", test_ids)\n'
    'train_ds.ids = train_ds.ids[::180][:8]; val_ds.ids = val_ds.ids[::60][:12]; test_ds.ids = test_ds.ids[::60][:12]\n')

e2 = os.path.join(work, "outputs", "E2_safety"); e1 = os.path.join(work, "outputs", "E1_detectors")
os.makedirs(e2, exist_ok=True); os.makedirs(e1, exist_ok=True)
shutil.copy(os.path.join(PROJ, "results", "E2", "split.json"), e2)
shutil.copy(os.path.join(PROJ, "results", "E1", "E1_summary.json"), e1)
if len(sys.argv) > 1:
    src += chr(10) + open(sys.argv[1], encoding="utf-8").read()
exec(compile(src, "E3_zero_shot.py", "exec"), {"__name__": "__main__"})
print("HARNESS: completed without errors")
