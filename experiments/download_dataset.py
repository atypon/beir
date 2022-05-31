import os
import pathlib
from typing import List
from beir import util


def download_datasets(datasets: List[str], datasets_path: str) -> None:
    if not os.path.isdir(datasets_path):
        os.mkdir(datasets_path)
    for dataset in datasets:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        util.download_and_unzip(url, os.path.join(datasets_path, dataset))
