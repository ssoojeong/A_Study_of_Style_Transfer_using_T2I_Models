from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import argparse
import os
import cv2
import numpy as np
import torch

parser = argparse.ArgumentParser()
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
        "--gpu",
        type=int,
        help="path to prompt_dir"
    )

args = parser.parse_args()

torch.cuda.set_device(args.gpu)


pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# or
# euler_scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
# pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=euler_scheduler)


for file in sorted(os.listdir(args.prompt_dir)):
    with open(os.path.join(args.prompt_dir, file), 'r') as f:
        prompt = f.readlines()[0]
        print(prompt)
        name = file.split('.')[0]
        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(os.path.join(args.save_dir,f"{name}.png"))    







