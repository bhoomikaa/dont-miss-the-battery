# Local CPU tests

Before each GPU run, the experiment notebooks were run end to end on a laptop CPU with a tiny subset of
RecyBat24 (8 training, 12 validation and 12 to 28 test images, 1 epoch). These scripts drive those runs, and
`logs/` holds their output.

| Script | What it does | Log |
|---|---|---|
| `harness.py` + `checks_e2.py` | Runs the E2 notebook on CPU, then the synthetic metric checks: perfect predictions must give mAP 1.0 and zero misses; correctly placed boxes with wrong labels must count as found (class-agnostic) and missed (class-aware) | `logs/E2_cpu_run_and_synthetic_checks.txt` |
| `harness_e1.py` + `checks_e1.py` | Runs the E1 notebook (YOLO11-s, RT-DETR-l) on CPU, then converts the YOLO label files back into predictions, which must score mAP 1.0 | `logs/E1_cpu_run_and_roundtrip_check.txt` |
| `harness_e1_resume.py` | Forced-disconnect test: run 1 is killed after epoch 1, run 2 must resume at epoch 2 and finish | `logs/E1_resume_step1_forced_disconnect.txt`, `logs/E1_resume_step2_resumed.txt` |
| `harness_e45.py` | Runs the E4/E5 notebook on CPU with stand-in weights (metrics are meaningless; this checks that every cell runs) | `logs/E4_E5_cpu_run.txt` |
| `harness_e3.py` | Runs the E3 notebook on CPU with the three zero-shot models | `logs/E3_cpu_run.txt` |

Usage example: `python harness.py E2_safety_training.py <path to recybat24> checks_e2.py`.
Paths inside the scripts point to the machine the tests were run on.
