#!/bin/bash

mkdir model
mkdir model/miniLM
gsutil cp -r gs://ai_repo/specter/knowledge-distillation/all-MiniLM-L12-v2/tokenizer/tokenizer/* model/miniLM/
gsutil cp gs://ai_repo/specter/knowledge-distillation/all-MiniLM-L12-v2/all_minilm_l12_fp16_optimized.onnx model/miniLM/