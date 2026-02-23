from .chatgpt import ChatGPT
from .alpaca import Alpaca
from .longchat.longchat import Longchat
from .base_language_model import BaseLanguageModel
from .llama import Llama
from .flan_t5 import FlanT5

try:
    from .vllm import VLLM
    _VLLM_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - optional dependency
    VLLM = None  # type: ignore
    _VLLM_AVAILABLE = False
    _VLLM_IMPORT_ERROR = _exc

_registered_pairs = []
if _VLLM_AVAILABLE:
    _registered_pairs.append(('vllm', VLLM))
_registered_pairs.extend(
    [
        ('gpt-4', ChatGPT),
        ('gpt-3.5-turbo', ChatGPT),
        ('alpaca', Alpaca),
        ('longchat', Longchat),
        ('llama', Llama),
        ('flan-t5', FlanT5),
        ('rog', Llama),
    ]
)
registed_language_models = dict(_registered_pairs)

def get_registed_model(model_name) -> BaseLanguageModel:
    if 'vllm' in model_name.lower() and not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed. Please install it to use --model_name vllm."
        ) from _VLLM_IMPORT_ERROR
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    raise ValueError(f"No registered model found for name {model_name}")
