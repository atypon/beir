import numpy as np
import onnx
import torch
from typing import Tuple
from onnxruntime import InferenceSession
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers import optimizer
from transformers import AutoModel, AutoTokenizer


class OnnxConverter:
    """
    Class that implements an ONNX converter for hugginface models
    """

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_len: int,
        device: str = 'cpu',
    ) -> None:
        """
        :param model: The pretrained model which is fine-tuned
        :param tokenizer: An object to tokenize the input text
        :param max_len: the maximum length of the sequences that the model
            can handle
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.model.to(self.device)

    def __create_input(self) -> Tuple[torch.LongTensor, ...]:
        """
        Tokenize the given text and create the inputs for the conversion
        to ONNX
        :return: The input tensors to convert the model to ONNX
        """
        dummy_text = """
        This is an example text. Lorem ipsum is a placeholder text commonly
        used to demonstrate the visual form of a document or a typeface""" * 50
        tokens = self.tokenizer.encode_plus(
            text=dummy_text,
            text_pair=dummy_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        inputs = (torch.LongTensor(tokens['input_ids']).to(self.device),
                  torch.LongTensor(tokens['attention_mask']).to(self.device))
        return tuple(inputs)

    @torch.inference_mode()
    def convert_to_onnx(self, path: str, opset_version: int) -> None:
        """
        Convert the model to ONNX
        :param path: the path to store the onnx model
        :param opset_version: the ONNX opset version to use
        """
        self.model.eval()
        input_names = ['input_ids', 'attention_mask']
        output_names = ['logits']
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        dynamic_axes = {
                'input_ids': symbolic_names,
                'attention_mask': symbolic_names,
                'logits': symbolic_names}
        inputs = self.__create_input()
        torch.onnx.export(
            self.model,
            args=inputs,
            f=path,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        # Better check from path to avoid large model errors
        # (also suggested by documentation)
        onnx.checker.check_model(path)

    def convert_to_fp16(
        self,
        path: str,
        fp16_path: str,
        num_heads: int = 12,
        hidden_size: int = 312
    ) -> None:
        """
        Convert to fp16
        :param path: path to onnx fp32 model
        :param fp16_path: path to save fp16 model
        :param int num_heads: number of mha heads
        :param int hidden_size: hidden state size of transformer
        """
        opt_options = optimizer.FusionOptions('bert')
        opt_options.enable_gelu_approximation = True
        opt_model = optimizer.optimize_model(
            path,
            model_type='bert',
            use_gpu=True,
            opt_level=1,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=opt_options
        )
        opt_model.convert_float_to_float16()
        opt_model.save_model_to_file(fp16_path)

    def optimize_fp_16(self, path: str, optimized_path: str) -> None:
        """
        Convert to optimized fp16
        :param path: the path to where fp16 is stored the onnx model
        :param optimized_path: the path of the optimized fp16
        """
        onnx.save(
            SymbolicShapeInference.infer_shapes(
                onnx.load(path), auto_merge=True
            ),
            optimized_path
        )

    def test_conversion(self, destination_path: str, decimals: int) -> None:
        """
        Test for successful convertion
        :param destination_path: onnx file to check
        :param decimal: number of decimals to check
        """
        inputs = self.__create_input()
        # Infer actual model
        golden_logits = self.model(*inputs).logits.detach().cpu().numpy()
        # Infer Onxx model
        session = InferenceSession(
            path_or_bytes=destination_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        logits = session.run(
            None,
            input_feed={
                'input_ids': inputs[0].detach().cpu().numpy(),
                'attention_mask': inputs[1].detach().cpu().numpy()
                }
        )[0]
        np.testing.assert_array_almost_equal(
            x=golden_logits,
            y=logits,
            decimal=decimals
        )
