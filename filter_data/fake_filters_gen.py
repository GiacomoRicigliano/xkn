import sys

import numpy as np

if len(sys.argv) == 2:
    fname = sys.argv[1]
else:
    fname = "iso_calc.py"

# in nano meters
filters = np.geomspace(2, 300000, 5000)
filters = np.unique(np.round(filters))

with open(fname, "w") as f:
    f.write("import json\n")
    f.write("\n")
    f.write("filters = {}")
    f.write("\n")
    for filt in filters:
        filt_lam = str(int(filt * 1000) / 1000)
        filt = str(int(filt * 1000))
        f.write(
            "filters["
            + filt
            + "] = { 'name'  : 'fake"
            + filt
            + "', 'lambda': "
            + filt_lam
            + "e-9, 'type': 'AB', 'filename': ['mag_CTIO_band_B.txt']}\n"
        )
    f.write("\n")
    f.write("with open('iso_calc.json', 'w') as fi:\n")
    f.write("    json.dump(filters, fi, indent = '    ', sort_keys = True)")
