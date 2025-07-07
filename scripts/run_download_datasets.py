import argparse
import os

from beir import util

from beir_extensions.configs import load_configurations


def download_datasets(datasets: list[str], datasets_path: str) -> None:
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
    arg_parser.add_argument(
        '--config',
        '-cf',
        help='Config file that contains the datasets to download.'
    )
    args = arg_parser.parse_args()

    cfg = load_configurations(path=args.config)
    download_datasets(
        datasets=[dataset for dataset in cfg.datasets],
        datasets_path='datasets'
    )
