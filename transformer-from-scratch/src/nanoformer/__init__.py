"""nanoformer: a decoder-only Transformer built from first principles in PyTorch.

Public surface (stable for the tests and the CLI):
    ModelConfig, Transformer, KVCache, BPETokenizer, Trainer, TrainConfig, generate
"""

from nanoformer.cache import KVCache
from nanoformer.config import ModelConfig
from nanoformer.generate import generate
from nanoformer.model import Transformer
from nanoformer.tokenizer import BPETokenizer
from nanoformer.trainer import TrainConfig, Trainer

__all__ = [
    "BPETokenizer",
    "KVCache",
    "ModelConfig",
    "TrainConfig",
    "Trainer",
    "Transformer",
    "generate",
]

__version__ = "0.1.0"
