import numpy as np
from PIL import Image

import gradio as gr
import re
import bleach
import sys
import os


import cv2
import argparse
import torch
import transformers
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPImageProcessor

from model.LISA import LISA
from utils.conversation import get_default_conv_template
from model.segment_anything.utils.transforms import ResizeLongestSide

def parse_args(args):
  parser = argparse.ArgumentParser(description='LISA chat')
  parser.add_argument('--version', default='./LISA-13B-llama2-v0-explainatory')
  parser.add_argument('--vis_save_path', default='./vis_output', type=str)
  parser.add_argument('--precision', default='fp16', type=str, choices=['fp32', 'bf16', 'fp16'], help="precision for inference")
  parser.add_argument('--image-size', default=1024, type=int, help='image size')
  parser.add_argument('--model-max-length', default=512, type=int)
  parser.add_argument('--lora-r', default=-1, type=int)
  parser.add_argument('--vision-tower', default='openai/clip-vit-large-patch14', type=str)
  parser.add_argument('--local-rank', default=0, type=int, help='node rank')
  parser.add_argument('--load_in_8bit', action='store_true', default=False)
  parser.add_argument('--load_in_4bit', action='store_true', default=True)
  return parser.parse_args(args)

def preprocess(x, 
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), 
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024
  ) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens('[SEG]')
ret_token_idx = tokenizer('[SEG]', add_special_tokens=False).input_ids
args.seg_token_idx = ret_token_idx[0]

model = LISA(
args.local_rank, 
args.seg_token_idx, 
tokenizer, 
args.version, 
args.lora_r,
args.precision,
load_in_8bit=args.load_in_8bit,
load_in_4bit=args.load_in_4bit,
)

weight = {}
visual_model_weight = torch.load(os.path.join(args.version, "pytorch_model-visual_model.bin"))
text_hidden_fcs_weight = torch.load(os.path.join(args.version, "pytorch_model-text_hidden_fcs.bin"))
weight.update(visual_model_weight)
weight.update(text_hidden_fcs_weight)
missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)

if args.precision == 'bf16':
    model = model.bfloat16().cuda()
elif args.precision == 'fp16':
    import deepspeed
    model_engine = deepspeed.init_inference(model=model, 
        dtype=torch.half, 
        replace_with_kernel_inject=True,
        replace_method="auto",
    )
    model = model_engine.module
else:
    model = model.float().cuda()

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
image_token_len = 256

clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
transform = ResizeLongestSide(args.image_size)


# Gradio
examples = [
    ['Where can the driver see the car speed in this image? Please output segmentation mask.', './resources/imgs/example1.jpg'],
    ['Can you segment the food that tastes spicy and hot?', './resources/imgs/example2.jpg'],
    ['Assuming you are an autonomous driving robot, what part of the diagram would you manipulate to control the direction of travel? Please output segmentation mask and explain why.', './resources/imgs/example1.jpg'],
    ['What can make the woman stand higher? Please output segmentation mask and explain why.', './resources/imgs/example3.jpg'],
]
output_labels = ['Segmentation Output']

title = 'LISA: Reasoning Segmentation via Large Language Model'

description = """
<font size=4>
This is the online demo of LISA. \n
If multiple users are using it at the same time, they will enter a queue, which may delay some time. \n
**Note**: **Different prompts can lead to significantly varied results**. \n
**Note**: Please try to **standardize** your input text prompts to **avoid ambiguity**, and also pay attention to whether the **punctuations** of the input are correct. \n
**Note**: Current model is **LISA-13B-llama2-v0-explanatory**, and 4-bit quantization may impair text-generation quality. \n
**Usage**: <br>
&ensp;(1) To let LISA **segment something**, input prompt like: "Can you segment xxx in this image?", "What is xxx in this image? Please output segmentation mask."; <br>
&ensp;(2) To let LISA **output an explanation**, input prompt like: "What is xxx in this image? Please output segmentation mask and explain why."; <br>
&ensp;(3) To obtain **solely language output**, you can input like what you should do in current multi-modal LLM (e.g., LLaVA). <br>

Hope you can enjoy our work!
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2308.00692' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/dvlab-research/LISA' target='_blank'>   Github Repo </a></p>
"""


## to be implemented
def inference(input_str, input_image):

    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    ## input valid check
    if not re.match(r'^[A-Za-z ,.!?\'\"]+$', input_str) or len(input_str) < 1:
        output_str = '[Error] Invalid input: ', input_str
        # output_image = np.zeros((128, 128, 3))
        ## error happened
        output_image = cv2.imread('./resources/error_happened.png')[:,:,::-1]
        return output_image, output_str

    # Model Inference
    conv = get_default_conv_template("vicuna").copy()
    conv.messages = []

    prompt = input_str #input("Please input your prompt: ")
    prompt = DEFAULT_IMAGE_TOKEN + " " + prompt
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # image_path = input("Please input the image path: ")
    # if not os.path.exists(image_path):
    #     print("File not found in {}".format(image_path))
    #     continue
    # image = cv2.imread(image_path)
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = input_image

    original_size_list = [image.shape[:2]]
    if args.precision == 'bf16':
        images_clip = clip_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().bfloat16()
    elif args.precision == 'fp16':
        images_clip = clip_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().half()
    else:
        images_clip = clip_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).cuda().float()
    images = transform.apply_image(image)
    resize_list = [images.shape[:2]]
    if args.precision == 'bf16':
        images = preprocess(torch.from_numpy(images).permute(2,0,1).contiguous()).unsqueeze(0).cuda().bfloat16()
    elif args.precision == 'fp16':
        images = preprocess(torch.from_numpy(images).permute(2,0,1).contiguous()).unsqueeze(0).cuda().half()
    else:
        images = preprocess(torch.from_numpy(images).permute(2,0,1).contiguous()).unsqueeze(0).cuda().float()

    input_ids = tokenizer(prompt).input_ids
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
    output_ids, pred_masks = model.evaluate(images_clip, images, input_ids, resize_list, original_size_list, max_new_tokens=512, tokenizer=tokenizer)
    text_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    text_output = text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "").replace("\n", "").replace("  ", "").replace("</s>", "")
    text_output = text_output.split("ASSISTANT: ")[-1]

    print("text_output: ", text_output)
    save_img = None
    for i, pred_mask in enumerate(pred_masks):

        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = (pred_mask > 0)

        # save_path = "{}/{}_mask_{}.jpg".format(args.vis_save_path, image_path.split("/")[-1].split(".")[0], i)
        # cv2.imwrite(save_path, pred_mask * 100)
        # print("{} has been saved.".format(save_path))
        
        # save_path = "{}/{}_masked_img_{}.jpg".format(args.vis_save_path, image_path.split("/")[-1].split(".")[0], i)
        save_img = image.copy()
        save_img[pred_mask] = (image * 0.5 + pred_mask[:,:,None].astype(np.uint8) * np.array([255,0,0]) * 0.5)[pred_mask]
        # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path, save_img)
        # print("{} has been saved.".format(save_path))
        # height = 360
        # width = int(save_img.shape[1] / save_img.shape[0] * height)
        # save_img = cv2.resize(save_img, dsize=(width, height))

    output_str = 'ASSITANT: ' + text_output #input_str
    if save_img is not None:
        output_image = save_img #input_image
    else:
        ## no seg output
        output_image = cv2.imread('./resources/no_seg_out.png')[:,:,::-1]
    return output_image, output_str




demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(
            lines=1, placeholder=None, label='Text Instruction'),
        gr.Image(type='filepath', label='Input Image'),
    ],
    outputs=[
        gr.Image(type="pil", label='Segmentation Output'),
        gr.Textbox(
            lines=1, placeholder=None, label='Text Output'),        
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging='auto',
    flagging_dir='/data/lisa_flagging_data')

demo.queue()

demo.launch()