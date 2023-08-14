import numpy as np
from PIL import Image

import gradio as gr
import re
import bleach
import sys
import os

from openxlab.model import download
download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-text_hidden_fcs.bin', output='./LISA-13B-llama2-v0-explainatory')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00001-of-00003.bin', output='./LISA-13B-llama2-v0-explainatory')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00002-of-00003.bin', output='./LISA-13B-llama2-v0-explainatory')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00003-of-00003.bin', output='./LISA-13B-llama2-v0-explainatory')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-visual_model.bin', output='./LISA-13B-llama2-v0-explainatory')

print(args.version)

if not os.path.exists('./bitsandbytes'):
    os.system("git clone https://github.com/timdettmers/bitsandbytes.git && cd bitsandbytes && CUDA_VERSION=113 make cuda11x && python setup.py install && cd .. && rm -r bitsandbytes && export PYTHONPATH=./ && python3 run.py")
