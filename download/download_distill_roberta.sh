#!/bin/bash
gsutil cp gs://ai_repo/specter/original_data/models/distilroberta_tuned/dim_reduction_models/1.9/distilroberta_multi_pca_278_fp16_optimized.onnx model/
gsutil cp -r gs://ai_repo/specter/original_data/models/distilroberta_tuned/tokenizer/* model/
