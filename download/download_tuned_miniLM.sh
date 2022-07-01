#!/bin/bash

mkdir model/tuned_miniLM
gsutil cp gs://ai_repo/specter/semantic_scholar_dataset_2hops/specter_like/all_minilm_l12_v2_multi_objective/all_minilm_l12_v2_pca_256_multi_objective_fp16_optimized.onnx model/tuned_miniLM/
gsutil cp gs://ai_repo/specter/semantic_scholar_dataset_2hops/specter_like/all_minilm_l12_v2_multi_objective/tokenizer_config.json model/tuned_miniLM/
gsutil cp gs://ai_repo/specter/semantic_scholar_dataset_2hops/specter_like/all_minilm_l12_v2_multi_objective/tokenizer.json model/tuned_miniLM/
gsutil cp gs://ai_repo/specter/semantic_scholar_dataset_2hops/specter_like/all_minilm_l12_v2_multi_objective/vocab.txt model/tuned_miniLM/
gsutil cp gs://ai_repo/specter/semantic_scholar_dataset_2hops/specter_like/all_minilm_l12_v2_multi_objective/special_tokens_map.json.json model/tuned_miniLM/