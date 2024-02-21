import argparse
import os
from typing import List
import yaml

from beir import util


def download_datasets(datasets: List[str], datasets_path: str) -> None:
    """
    Function for downloading selected datasets
    :param datasets: list of dataset name to download
    :param datasets_path: destination folder
    """
    if not os.path.isdir(datasets_path):
        os.mkdir(datasets_path)
    for dataset in datasets:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        util.download_and_unzip(url, datasets_path)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config',
                            '-cf',
                            help='The path of the config file that contains the datasets for download')
    args = arg_parser.parse_args()

    with open(args.config) as config_file:
        dataset_configs = yaml.safe_load(config_file)
    download_datasets(datasets=dataset_configs['datasets'], datasets_path='datasets')
