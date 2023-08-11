import numpy as np
from PIL import Image

import gradio as gr
import re
import bleach
import sys
import os

if not os.path.exists('./bitsandbytes'):
    os.system("git clone https://github.com/timdettmers/bitsandbytes.git && cd bitsandbytes && CUDA_VERSION=113 make cuda11x && python setup.py install && cd .. && rm -r bitsandbytes && export PYTHONPATH=./ && python3 run.py && python git lfs install && git lfs clone https://huggingface.co/spaces/xinlai/LISA/tree/main/LISA-13B-llama2-v0-explainatory")

