import yaml
from experiments.download_dataset import download_datasets

if __name__ == '__main__':
    with open('configs/miniLM/semantic_search_config.yaml') as config_file:
        dataset_configs = yaml.safe_load(config_file)
    download_datasets(datasets=dataset_configs['datasets'], datasets_path=dataset_configs['datasets_path'])
