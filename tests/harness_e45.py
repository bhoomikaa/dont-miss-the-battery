"""CPU smoke test for E4_E5_robustness_gradcam.py: tiny subset, severities 1 and 5, stand-in weights."""
import os, sys, re, shutil, matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = "C:/Users/bhoom/OneDrive/Desktop/dont-miss-the-battery"
work = os.path.join(HERE, "run_E45"); os.makedirs(work, exist_ok=True); os.chdir(work)
src = open(os.path.join(PROJ, "E4_E5_robustness_gradcam.py"), encoding="utf-8").read()
def sub(old, new):
    global src
    assert old in src, old
    src = src.replace(old, new)
root = "C:/Users/bhoom/datasets/recybat24/recybat24"
open(os.path.join(root, ".done"), "w").close()
sub('ROOT = "/content/recybat24"', f'ROOT = "{root}"')
sub("!pip -q install ultralytics==8.4.216", "pass")
sub("SEVERITIES = [1, 2, 3, 4, 5]", "SEVERITIES = [1, 3, 5]")
src = re.sub(r"^N_BOOT = \d+", "N_BOOT = 50", src, flags=re.M)
sub('test_ds = BatteryDet("val", test_ids)\n',
    'test_ds = BatteryDet("val", test_ids)\n'
    'train_ds.ids = train_ds.ids[::180][:8]; val_ds.ids = val_ds.ids[::60][:12]; test_ds.ids = test_ds.ids[::25][:28]\n')

# stand-in Drive folders
e2 = os.path.join(work, "outputs", "E2_safety"); e1 = os.path.join(work, "outputs", "E1_detectors")
if not os.path.exists(os.path.join(e2, "repeat_factor_best.pt")):
    os.makedirs(e2, exist_ok=True)
    shutil.copy(os.path.join(PROJ, "results", "E2", "split.json"), e2)
    import torch, torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    torch.manual_seed(0)
    m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=640, max_size=640)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 4)
    for n in ["baseline_best.pt", "repeat_factor_best.pt"]:
        torch.save(m.state_dict(), os.path.join(e2, n))
    for n in ["yolo11s", "rtdetr_l"]:
        os.makedirs(os.path.join(e1, n, "weights"), exist_ok=True)
        shutil.copy(os.path.join(HERE, "run_E1", "outputs", "E1_detectors", n, "weights", "best.pt"), os.path.join(e1, n, "weights"))
    shutil.copy(os.path.join(PROJ, "results", "E1", "E1_summary.json"), e1)
if len(sys.argv) > 1:
    src += chr(10) + open(sys.argv[1], encoding="utf-8").read()
exec(compile(src, "E4_E5_robustness_gradcam.py", "exec"), {"__name__": "__main__"})
print("HARNESS: completed without errors")
