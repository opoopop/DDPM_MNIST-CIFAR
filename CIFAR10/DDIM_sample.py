import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import os
import argparse
from torch.optim import Adam,AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from diffusers import DDIMPipeline




def count_images_in_path(directory):
    """
    统计指定路径及其子目录下的图片文件数量。

    Args:
        directory (str): 文件夹路径。

    Returns:
        int: 图片文件的总数量。
    """
    # 支持的图片文件扩展名
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    
    # 初始化图片计数
    image_count = 0

    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为有效图片格式
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_count += 1
    
    return image_count
# load mdoel
device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
model1.to(device)
model1.eval()



model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x0_5", pretrained=True)
model2.to(device)
model2.eval()


model3 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5", pretrained=True)
model3.to(device)
model3.eval()

model4 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model4.to(device)
model4.eval()



model_list=[model1,model2,model3,model4]

# 加载预训练的 DDPM 模型
pipe = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

cnt=[0]*5
for i in range(5):
    cnt[i]=count_images_in_path(f'/home/chunjie/DdpmClassifierTest/CIFAR10/generated_images/image{i}')
    print(f'already have {cnt[i]} images')



def image_select(input_tensor):

    last=-1
    min_prob=2
    for model in model_list:
        with torch.no_grad():
            output = model(input_tensor)

        probabilities = F.softmax(output, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()

        if predicted_class>=5:
            return False,predicted_class
        if last==-1:
            last=predicted_class
        elif last!=predicted_class :
            return False,predicted_class
        

        min_prob=min(min_prob,max_prob)
        if min_prob<0.5:
            return False,predicted_class
        
    return True,predicted_class
        




preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像尺寸为 32x32
    transforms.ToTensor(),        # 转换为 Tensor，并将值归一化到 [0, 1]
    transforms.Normalize(         # 标准化
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 数据集的均值
        std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 数据集的标准差
    )
])

turn =0 
num_all=1000
while True:
    turn+=1
    print(f'start turn {turn}')
    total=0
    num_image=300
    images = pipe(batch_size=num_image).images
    for i in range(num_image):
        input_tensor = preprocess(images[i]).unsqueeze(0)  # 增加 batch 维度
        input_tensor = input_tensor.to(device)         # 将数据移动到设备（GPU 或 CPU）
        ff, label=image_select(input_tensor)
        label=label.item()
        if ff==True and cnt[label]<num_all:
            images[i].save(f'/home/chunjie/DdpmClassifierTest/CIFAR10/generated_images/image{label}/image_{cnt[label]}.png')
            cnt[label]+=1
            total+=1
            
    print(f'{total} number of images is generated in this turn')
    print('each of the class have number: ')
    ft =1
    for i in range(5):
        print(f'class {i}: {cnt[i]}')
        if cnt[i]<num_all:
            ft=0
    if ft == 1:  
        break





