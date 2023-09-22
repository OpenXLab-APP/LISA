import numpy as np
from PIL import Image

import gradio as gr
import re
import bleach
import sys
import os

from huggingface_hub import snapshot_download
snapshot_download(repo_id="xinlai/LISA-13B-llama2-v0-explanatory",  allow_patterns='*.bin', local_dir='./LISA-13B-llama2-v0-explainatory')

if not os.path.exists('./bitsandbytes'):
    os.system("git clone https://github.com/timdettmers/bitsandbytes.git && cd bitsandbytes && CUDA_VERSION=113 make cuda11x && python setup.py install && cd .. && rm -r bitsandbytes && export PYTHONPATH=./ && python3 run.py")
