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
parser.add_argument(
        "--spe_step",
        type=int,
        default=None,
        help="path to specific_step"
    )


args = parser.parse_args()

model_id = "CompVis/stable-diffusion-v1-4"

os.makedirs(args.save_dir, exist_ok=True)

if args.spe_step:
    spe_step_dir = os.path.join(args.model_dir, f'checkpoint-{str(args.spe_step)}')
    unet = UNet2DConditionModel.from_pretrained(os.path.join(spe_step_dir, 'unet'))
else:
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.model_dir, 'unet'))
text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.model_dir, 'text_encoder'))

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

for file in sorted(os.listdir(args.prompt_dir)):
    print(file)
    with open(os.path.join(args.prompt_dir, file), 'r') as f:
        prompt = f.readlines()[0]
        print(prompt)
        name = file.split('.')[0]
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(os.path.join(args.save_dir,f"{name}.png"))    

