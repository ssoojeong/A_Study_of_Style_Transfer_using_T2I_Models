import argparse
import os

original_path = os.getcwd()

def run_textual_inversion(args):
    os.chdir(os.path.join(original_path, f'models/{args.model}'))
    args.instance_dir = os.path.join(original_path, args.instance_dir)
    args.output_dir = os.path.join(original_path, args.output_dir, args.model)
    
    #initial prompts
    if 'peanuts' in args.instance_dir:
        initializer_token = 'sketch'
    else:
        initializer_token = 'painting'
    
    os.system(f"""
    accelerate launch textual_inversion.py \
      --pretrained_model_name_or_path={args.model_name} \
      --train_data_dir={args.instance_dir} \
      --learnable_property="style" \
      --placeholder_token="<new1>" --initializer_token="{initializer_token}" \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --max_train_steps=3000 \
      --learning_rate=5.0e-04 --scale_lr \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --output_dir={args.output_dir}
    """)
    
    os.chdir(original_path)
    

def run_dreambooth(args):
    os.chdir(os.path.join(original_path, f'models/{args.model}'))
    args.instance_dir = os.path.join(original_path, args.instance_dir)
    args.output_dir = os.path.join(original_path, args.output_dir, args.model)
    args.class_dir = os.path.join(original_path, args.class_dir)
    
    os.system(f"""
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path={args.model_name} \
      --instance_data_dir={args.instance_dir} \
      --class_data_dir={args.class_dir} \
      --output_dir={args.output_dir} \
      --with_prior_preservation --prior_loss_weight=1.0 \
      --instance_prompt="A painting in the style of <new1>" \
      --class_prompt="painting" \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --learning_rate=5e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=3000 \
      --num_train_epochs=100
    """)

def run_custom_diffusion(args):
    os.chdir(os.path.join(original_path, f'models/{args.model}'))
    args.instance_dir = os.path.join(original_path, args.instance_dir)
    args.output_dir = os.path.join(original_path, args.output_dir, args.model)
    args.class_dir = os.path.join(original_path, args.class_dir)
    
    os.system(f"""
    accelerate launch train_custom_diffusion.py \
      --pretrained_model_name_or_path={args.model_name} \
      --instance_data_dir={args.instance_dir} \
      --output_dir={args.output_dir} \
      --class_data_dir={args.class_dir} \
      --class_prompt="painting" --num_class_images=200 \
      --real_prior --prior_loss_weight=1.0 \
      --instance_prompt="A painting in the style of <new1>" \
      --resolution=512 \
      --train_batch_size=1 \
      --learning_rate=1e-5 \
      --lr_warmup_steps=0 \
      --max_train_steps=3000 \
      --scale_lr --hflip --noaug \
      --modifier_token="<new1>" \
      --num_train_epochs=100 \
      --no_safe_serialization
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True, choices=["textual_inversion", "dreambooth", "custom_diffusion"])
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--class_dir", type=str, default="data/sketch/images")
    parser.add_argument("--instance_dir", type=str, default="data/peanuts")
    parser.add_argument("--output_dir", type=str, default="save/images")

    args = parser.parse_args()

    # 모델 선택에 따라 실행
    if args.model == "textual_inversion":
        print("Running Textual Inversion...")
        run_textual_inversion(args)
    elif args.model == "dreambooth":
        print("Running DreamBooth...")
        run_dreambooth(args)
    elif args.model == "custom_diffusion":
        print("Running Custom Diffusion...")
        run_custom_diffusion(args)
    else:
        print("이 중에서 다시 입력: 'textual_inversion', 'dreambooth', or 'custom_diffusion'.")
