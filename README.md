# rnd-beir

This repository consists an extension to the main [beir repository](https://github.com/atypon/beir) providing us the needed extensions for evaluating our embedding models.

## Installation

For this extension, the main beir is installed as a library. To set up run the following:

```bash
git clone https://github.com/atypon/beir.git
conda create -y --name beir python=3.11
conda activate beir
pip3 install -e . --index-url https://download.pytorch.org/whl/cu126
```

To enable `flash-attn` then run `pip install flash-attn==2.7.4.post1 --no-build-isolation`

## Scripts

- `run_onnx_conversion.py` : Convert the specified model to onnx format.
- `run_download_datasets.py`: After pointing to a config file that contains the desired datasets, it downloads them.
- `run_dense_retrieval_experiment.py` : Performs dense retrieval evaluation od the desired datasets with the speficied model. Model can be `ONNXModel` or `SentenceTransformerModel`. By subclassing from `CustomModel` in `beir_extensions/models`, aby desired behaviours can be achieved. Check `dense_retrieval_experiment.yaml` and `dense_retrieval_experiment_onnx.yaml` for setting up the experiment properly.


## Legacy

Legacy scripts and configuration files have been put inside `legacy` folders. These files might have potential future value.
