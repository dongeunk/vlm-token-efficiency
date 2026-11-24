"""Run Study4's synthetic and tiny-model tests on CPU, without model downloads."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="-1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tests")]
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    torch.set_num_threads(1)

    def disabled(*args,**kwargs):
        raise RuntimeError("GPU and pretrained model/tokenizer loading are disabled for this CPU suite.")

    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    with tempfile.TemporaryDirectory(prefix="study4-cpu-") as scratch, \
            patch.object(tempfile,"tempdir",scratch), \
            patch.object(torch.cuda,"_lazy_init",disabled), \
            patch.object(PreTrainedModel,"from_pretrained",disabled), \
            patch.object(PreTrainedTokenizerBase,"from_pretrained",disabled):
        tests=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_*_cpu.py")
        result=unittest.TextTestRunner(verbosity=1).run(tests)
    return 0 if result.wasSuccessful() and not result.skipped and not torch.cuda.is_initialized() else 1


if __name__=="__main__":
    raise SystemExit(main())
