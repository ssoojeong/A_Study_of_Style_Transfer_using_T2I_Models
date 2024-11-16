from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
import torch
import argparse
import os
import cv2
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''

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
#pipeline = DiffusionPipeline.from_pretrained("/userHome/userhome1/sojeong/style/sd-v1-4.ckpt", torch_dtype=torch.float16, local_files_only=True).to("cuda")

pipeline.load_textual_inversion(args.model_dir)



for file in sorted(os.listdir(args.prompt_dir)):
    print(file)
    with open(os.path.join(args.prompt_dir, file), 'r') as f:
        prompt = f.readlines()[0]
        print(prompt)
        name = file.split('.')[0]
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(os.path.join(args.save_dir,f"{name}.png"))    

