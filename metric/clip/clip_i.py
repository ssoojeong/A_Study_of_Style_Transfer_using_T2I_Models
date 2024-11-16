import os
import clip
import torch
# from torchvision.datasets import CIFAR100
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--save_dir",
        type=str,
        default='/userHome/userhome1/sojeong/style/0203_metric/results',
        help="path to save dataset"
    )
parser.add_argument(
        "--image_dir1",
        type=str,
        help="path to prompt_dir"
    )
parser.add_argument(
        "--image_dir2",
        type=str,
        help="path to image_dir"
    )
parser.add_argument('-c', '--gpu', default='', type=int,
                    help='GPU to use (leave blank for CPU only)')

# Load the model

def main(args):
    
    os.makedirs(args.save_dir, exist_ok=True) ##추가

    #gpu
    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-B/32', device)

    img_path1 = args.image_dir1
    img_path2 = args.image_dir2

    total = 0.0
    average = 0.0
    total_len = len(os.listdir(img_path1))
    print(len(os.listdir(img_path1)))
    print(len(os.listdir(img_path2)))
    assert len(os.listdir(img_path1)) == len(os.listdir(img_path2))

    for img_f1, img_f2 in zip(sorted(os.listdir(img_path1)), sorted(os.listdir(img_path2))):
        # print(img_f)
        # print(txt_f)
        
        image1 = Image.open(os.path.join(img_path1, img_f1))
        image2 = Image.open(os.path.join(img_path2, img_f2))
        
        
        image_input1 = preprocess(image1).unsqueeze(0).to(device)
        image_input2 = preprocess(image2).unsqueeze(0).to(device)
        
        
        # Calculate features
        with torch.no_grad():
            image_features1 = model.encode_image(image_input1)
            image_features2 = model.encode_image(image_input2)

        # Pick the top 5 most similar labels for the image
        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)
        similarity = image_features1 @ image_features2.T
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #values, indices = similarity[0].topk(2)
        values, indices = similarity[0].topk(1)
        print(values.item())
        # Print the result
        total += values.item()
        
    #    print(f'CLIP-T: {100 * values.item():.2f}%')
        with open(os.path.join(args.save_dir, 'CLIP-I.txt'), 'a') as f:
            f.write(str(values.item()))
            f.write('\n')

    average = total / total_len
    with open(os.path.join(args.save_dir, 'CLIP-I.txt'), 'a') as f:
        f.write(f'Average: {average}')
        f.write('\n')

    print(f'Average: {average}')



if __name__ == '__main__':
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)