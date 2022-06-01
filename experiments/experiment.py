import os
from typing import List
from onnxruntime import InferenceSession

class Experiment(object):
    def __init__(self, datasets: List[str], datasets_path: str):
        self.dataset_paths = []
        for dataset in datasets:
            dataset_folder = os.path.join(os.path.join(datasets_path, dataset),
                                          dataset)
            self.dataset_paths.append(dataset_folder)

    def __setup_experiment(self, onnx_model: InferenceSession):
        pass