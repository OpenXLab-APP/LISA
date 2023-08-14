import numpy as np
from PIL import Image

import gradio as gr
import re
import bleach
import sys
import os

if not os.path.exists('./bitsandbytes'):
    os.system("git clone https://github.com/timdettmers/bitsandbytes.git && cd bitsandbytes && CUDA_VERSION=113 make cuda11x && python setup.py install && cd .. && rm -r bitsandbytes && export PYTHONPATH=./ && python3 run.py")

import openxlab
openxlab.login(ak=<q9bpknwq8valkv1xmvno>, sk=<gr24e76lkxoa1n0medzykgeygwddlvoq9bym58zg>)

from openxlab.model import download
download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-text_hidden_fcs.bin')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00001-of-00003.bin')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00002-of-00003.bin')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-00003-of-00003.bin')

download(model_repo='openxlab-app/pytorch_model-text_hidden_fcs.bin', 
model_name='pytorch_model-visual_model.bin')
