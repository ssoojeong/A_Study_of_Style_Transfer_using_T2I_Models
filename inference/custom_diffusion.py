from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import argparse
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
        "--model_dir",
        type=str,
        help="path to save dataset"
    )
parser.add_argument(
        "--save_dir",
        type=str,
        help="path to save dataset"
    )
parser.add_argument(
        "--prompt_dir",
        type=str,
        help="path to prompt_dir"
    )


args = parser.parse_args()


pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs(args.model_dir, weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion(args.model_dir, weight_name="<new1>.bin")

gen_imglist = os.listdir(args.save_dir)

for file in sorted(os.listdir(args.prompt_dir)):
    print(file)
    with open(os.path.join(args.prompt_dir, file), 'r') as f:
        prompt = f.readlines()[0]
        print(prompt)
        name = file.split('.')[0]
        check_img_name = name+'.png'
    if check_img_name in gen_imglist:
        continue
    else:
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(os.path.join(args.save_dir,f"{name}.png"))    
