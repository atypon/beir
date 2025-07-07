import torch
from transformers import AutoModel, AutoTokenizer

from beir.extensions.configs import load_configurations
from beir.extensions.onnx_conversion import OnnxConverter


if __name__ == '__main__':

    cfg = load_configurations('configs/onnx_conversion.yaml')
    
    model = AutoModel.from_pretrained(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side='left')

    onnx_converter = OnnxConverter(
        model=model,
        tokenizer=tokenizer,
        max_len=cfg.max_length,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    onnx_converter.convert_to_onnx(
        path=cfg.onnx_file,
        opset_version=cfg.onnx_opset_version
    )
    onnx_converter.convert_to_fp16(
        path=cfg.onnx_file,
        fp16_path=cfg.onnx_file_fp16,
    )
