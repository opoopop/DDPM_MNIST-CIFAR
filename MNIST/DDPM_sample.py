import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

select_index = argparse.ArgumentParser(description="Class of image creating")

select_index.add_argument("--select_index", type=int, default=0,
                        help="Class of image creating")

args = select_index.parse_args()
se_index=args.select_index


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 2)
)
model.load_state_dict(torch.load(f'./MNIST/model/MNIST_{se_index}.pt'))
model.to(device)
diffusion = GaussianDiffusion(
    model,
    objective = 'pred_noise',
    image_size = 28,
    timesteps = 500    # number of steps
)
diffusion.to(device)



def sample(diffusion, save_dir="samples", sample_number=64,start_index=0):
    """ 生成样本并保存到指定目录 """
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成图片
    generated_images = diffusion.sample(batch_size=sample_number)
    imgs = generated_images.reshape(sample_number, 28, 28).to('cpu')

    # 逐张保存图片
    for i in range(sample_number):
        plt.imsave(os.path.join(save_dir, f"image_{start_index+i}.png"), imgs[i], cmap="gray")

    print(f"Saved {sample_number} images to {save_dir}")


save_dir=f'MNIST/generated_images/image{se_index}'
for i in range(0,1000,500):
    print(f'start at {i}, end at {i+500-1}')
    sample(diffusion, save_dir=save_dir, sample_number=500,start_index=i)
