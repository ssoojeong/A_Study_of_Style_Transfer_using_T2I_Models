import argparse
import os
# from inference.textual_inversion import *
# from inference.dreambooth import *
# from inference.custom_diffusion import *

original_path = os.getcwd()
os.chdir(os.path.join(original_path, 'inference'))

def run_custom_diffusion(args):
    args.model_dir = os.path.join(original_path, args.model_dir)
    args.save_dir = os.path.join(original_path, args.save_dir, args.model)
    args.prompt_dir = os.path.join(original_path, args.prompt_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    os.system(f"""
    python custom_diffusion.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}"
    """)

def run_dreambooth(args):
    args.model_dir = os.path.join(original_path, args.model_dir)
    args.save_dir = os.path.join(original_path, args.save_dir, args.model)
    args.prompt_dir = os.path.join(original_path, args.prompt_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    os.system(f"""
    python dreambooth.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}" \
      --spe_step {args.spe_step}
    """)

def run_textual_inversion(args):
    args.model_dir = os.path.join(original_path, args.model_dir)
    args.save_dir = os.path.join(original_path, args.save_dir, args.model)
    args.prompt_dir = os.path.join(original_path, args.prompt_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    os.system(f"""
    python textual_inversion.py \
      --model_dir "{args.model_dir}" \
      --save_dir "{args.save_dir}" \
      --prompt_dir "{args.prompt_dir}"
    """)

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["custom_diffusion", "dreambooth", "textual_inversion"])
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--spe_step", type=int, default=3000)

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
    
    os.chdir(original_path)