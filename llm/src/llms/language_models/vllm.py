from typing import List, Optional
import inspect

from transformers import AutoTokenizer

from .base_language_model import BaseLanguageModel


class VLLM(BaseLanguageModel):
    DTYPE = {
        "auto": "auto",
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model_path",
            type=str,
            help="Hugging Face model or local path",
            default="meta-llama/Llama-2-7b-chat-hf",
        )
        parser.add_argument("--max_new_tokens", type=int, default=512)
        parser.add_argument(
            "--dtype",
            choices=["auto", "fp16", "bf16", "fp32"],
            default="auto",
        )
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
        parser.add_argument("--max_model_len", type=int, default=None)
        parser.add_argument("--enforce_eager", action="store_true")
        parser.add_argument("--disable_custom_all_reduce", action="store_true")
        parser.add_argument("--trust_remote_code", action="store_true")
        parser.add_argument("--hf_token", type=str, default=None)
        parser.add_argument("--vllm_swap_space", type=float, default=4.0)
        parser.add_argument("--vllm_quiet", action="store_true")
        parser.add_argument("--vllm_disable_tqdm", action="store_true")

    def __init__(self, args):
        super().__init__(args)
        self.maximun_token = 4096 - 100
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None

    def load_model(self, **kwargs):
        return AutoTokenizer.from_pretrained(**kwargs, use_fast=False)

    def _init_tokenizer(self):
        kwargs = {"use_fast": False}
        if self.args.hf_token:
            kwargs["token"] = self.args.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, **kwargs)
        model_max_len = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_len, int) and model_max_len < 10**6:
            self.maximun_token = max(model_max_len - 100, 1)

    def tokenize(self, text):
        if self.tokenizer is None:
            self._init_tokenizer()
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover - guarded at runtime
            raise ImportError(
                "vLLM is not installed. Please install it to use --model_name vllm."
            ) from exc

        if self.tokenizer is None:
            self._init_tokenizer()

        llm_kwargs = {
            "model": self.args.model_path,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "dtype": self.DTYPE.get(self.args.dtype, self.args.dtype),
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
            "enforce_eager": self.args.enforce_eager,
            "disable_custom_all_reduce": self.args.disable_custom_all_reduce,
            "trust_remote_code": self.args.trust_remote_code,
            "swap_space": self.args.vllm_swap_space,
        }
        if self.args.hf_token:
            llm_kwargs["hf_token"] = self.args.hf_token
        if self.args.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.args.max_model_len
        llm_kwargs.update(model_kwargs)

        if self.args.vllm_quiet:
            llm_sig = inspect.signature(LLM.__init__)
            quiet_options = {
                "disable_log_stats": True,
                "disable_log_requests": True,
                "disable_progress_bar": True,
                "log_stats": False,
            }
            for name, value in quiet_options.items():
                if name in llm_sig.parameters:
                    llm_kwargs[name] = value

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(max_tokens=self.args.max_new_tokens)

    def _generate(self, prompts: List[str]):
        gen_kwargs = {}
        if self.args.vllm_disable_tqdm:
            gen_sig = inspect.signature(self.llm.generate)
            if "use_tqdm" in gen_sig.parameters:
                gen_kwargs["use_tqdm"] = False
            if "disable_tqdm" in gen_sig.parameters:
                gen_kwargs["disable_tqdm"] = True
        return self.llm.generate(prompts, self.sampling_params, **gen_kwargs)

    def generate_sentence(self, llm_input):
        outputs = self._generate([llm_input])
        return outputs[0].outputs[0].text  # type: ignore

    def generate_batch(self, prompts: List[str]) -> List[Optional[str]]:
        outputs = self._generate(prompts)
        results: List[Optional[str]] = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append(None)
        return results
