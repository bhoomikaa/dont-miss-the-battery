"""Run a Colab-style percent script end-to-end on CPU with a tiny subset of RecyBat24."""
import os, sys, re, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
work = os.path.join(HERE, "run_" + os.path.splitext(os.path.basename(sys.argv[1]))[0])
os.makedirs(work, exist_ok=True); os.chdir(work)
src = open(sys.argv[1], encoding="utf-8").read()

def sub(old, new, count=1):
    global src
    assert old in src, f"harness: pattern not found: {old!r}"
    src = src.replace(old, new) if count == 0 else src.replace(old, new, count)

root = sys.argv[2].replace("\\", "/")
open(os.path.join(root, ".done"), "w").close()
sub('ROOT = "/content/recybat24"', f'ROOT = "{root}"')
src = re.sub(r"^EPOCHS = \d+", "EPOCHS = 1", src, flags=re.M)
src = re.sub(r"^N_BOOT = \d+", "N_BOOT = 50", src, flags=re.M)
sub("num_workers=2", "num_workers=0", count=0)
sub('test_ds = BatteryDet("val", test_ids)\n',
    'test_ds = BatteryDet("val", test_ids)\n'
    'train_ds.ids = train_ds.ids[::180][:8]; val_ds.ids = val_ds.ids[::60][:12]; test_ds.ids = test_ds.ids[::60][:12]\n')
if len(sys.argv) > 3:
    src += "\n" + open(sys.argv[3], encoding="utf-8").read()
exec(compile(src, os.path.basename(sys.argv[1]), "exec"), {"__name__": "__main__"})
print("HARNESS: completed without errors")
