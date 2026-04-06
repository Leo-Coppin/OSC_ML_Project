import os
import re

#Windows path to Mordred package - adjust if needed
#mordred_path = r"C:\ProgramData\anaconda3\envs\osc_ml\Lib\site-packages\mordred"

#Linux path to Mordred package - adjust if needed
mordred_path = "/home/lemo/miniconda3/envs/osc_ml/lib/python3.11/site-packages/mordred"

# Remplacements nécessaires numpy 1.x -> 2.x
replacements = {
    "from numpy import product": "from numpy import prod as product",
    "numpy.product":             "numpy.prod",
    "np.product":                "np.prod",
    "from numpy import cumproduct": "from numpy import cumprod as cumproduct",
    "np.cumproduct":             "np.cumprod",
    "from numpy import sometrue": "from numpy import any as sometrue",
    "np.sometrue":               "np.any",
    "from numpy import alltrue":  "from numpy import all as alltrue",
    "np.alltrue":                "np.all",
    "from numpy import in1d":     "from numpy import isin as in1d",
    "np.in1d":                   "np.isin",
    "np.bool":                   "np.bool_",
    "np.int ":                   "np.int_ ",
    "np.float ":                 "np.float64 ",
    "np.complex ":               "np.complex128 ",
    "np.object ":                "np.object_ ",
    "np.str ":                   "np.str_ ",
}

patched_files = []

for root, dirs, files in os.walk(mordred_path):
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = content
        for old, new in replacements.items():
            new_content = new_content.replace(old, new)

        if new_content != content:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            patched_files.append(fname)

if patched_files:
    print(f"✅ {len(patched_files)} fichier(s) patché(s) :")
    for f in patched_files:
        print(f"   - {f}")
else:
    print("ℹ️  Aucun fichier à patcher.")