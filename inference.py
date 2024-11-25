import argparse
import os
from inference.textual_inversion import *
from inference.dreambooth import *
from inference.custom_diffusion import *

def run_custom_diffusion(args):
    os.system(f"""
    python custom_diffusion.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}"
    """)

def run_dreambooth(args):
    os.system(f"""
    python dreambooth.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}" \
      --spe_step {args.spe_step}
    """)

def run_textual_inversion(args):
    os.system(f"""
    python textual_inversion.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}"
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["custom_diffusion", "dreambooth", "textual_inversion"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--spe_step", type=int, default=None)

    args = parser.parse_args()

    # 모델 선택에 따라 다른 파이프라인 실행
    if args.model == "custom_diffusion":
        print("Running Custom Diffusion...")
        run_custom_diffusion(args)
    elif args.model == "dreambooth":
        print("Running DreamBooth...")
        run_dreambooth(args)
    elif args.model == "textual_inversion":
        print("Running Textual Inversion...")
        run_textual_inversion(args)
    else:
        print("이 중에서 선택: 'custom_diffusion', 'dreambooth', or 'textual_inversion'.")
