from transformers import pipeline, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')


    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs, use_fast=False, token="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")
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,  
        use_fast=False, token="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")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #model_kwargs.update({'use_auth_token': True})
        print("model: ", self.args.model_path)
        self.generator = pipeline("text-generation", token="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", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
        # 统一在 generation_config 中设置生成参数，避免重复参数告警与采样不稳定
        gen_cfg = getattr(self.generator, "generation_config", None)
        if gen_cfg is None:
            gen_cfg = self.generator.model.generation_config
        gen_cfg.max_new_tokens = self.args.max_new_tokens
        gen_cfg.max_length = None
        gen_cfg.do_sample = False
        gen_cfg.temperature = None
        if self.tokenizer.pad_token_id is not None:
            gen_cfg.pad_token_id = self.tokenizer.pad_token_id
            if getattr(self.generator.model, "config", None) is not None:
                self.generator.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, do_sample=False)
        return outputs[0]['generated_text'] # type: ignore

    @torch.inference_mode()
    def generate_sentences(self, llm_inputs, batch_size=4):
        outputs = self.generator(
            llm_inputs,
            return_full_text=False,
            do_sample=False,
            batch_size=batch_size,
        )
        results = []
        for out in outputs:
            if isinstance(out, list):
                results.append(out[0]["generated_text"])
            else:
                results.append(out["generated_text"])
        return results
