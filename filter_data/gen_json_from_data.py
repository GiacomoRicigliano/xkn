import os
import json
from copy import deepcopy
from pprint import pprint

import numpy as np

data_path = "AT2017gfo"
file_list = os.listdir(os.fsencode(data_path))

unique_lams = np.unique(
    np.array(
        [str(int(os.fsdecode(file)[:-4].split("_")[5]) // 10) for file in file_list]
    )
)

sub_dict = {
    "band": [],
    "filename": [],
    "lambdas": [],
    "name": [],
    "type": [],
}

filter_dict = {lam: deepcopy(sub_dict) for lam in unique_lams}

for file in os.listdir(os.fsencode(data_path)):
    filename = os.fsdecode(file)
    _, name, _, band, _, lambd, _, mtype = filename[:-4].split("_")
    lam = str(int(lambd) // 10)

    filter_dict[lam]["band"].append(band)
    filter_dict[lam]["filename"].append(filename)
    filter_dict[lam]["lambdas"].append(float(lambd) * 1e-10)
    filter_dict[lam]["name"].append(name)
    filter_dict[lam]["type"].append(mtype)
    filter_dict[lam]["lambda"] = np.mean(np.array(filter_dict[lam]["lambdas"]))

pprint(filter_dict)

with open(f"{data_path}.json", "w") as fi:
    json.dump(filter_dict, fi, indent="    ", sort_keys=True)
