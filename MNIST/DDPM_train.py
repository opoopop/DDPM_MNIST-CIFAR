import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import os
import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.optim import Adam,AdamW
from tqdm.auto import tqdm

select_index = argparse.ArgumentParser(description="Class of image creating")

select_index.add_argument("--select_index", type=int, default=0,
                        help="Class of image creating")

args = select_index.parse_args()
se_index=args.select_index
# 指定需要的类别
selected_classes = [se_index]
label_mapping = {orig_label: idx for idx, orig_label in enumerate(selected_classes)}





# 加载 CIFAR-100 数据集
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=False,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),  # 转换成张量
                                    ]))
# 筛选出所需类别的索引
selected_indices_train = [
    idx for idx, (_, label) in enumerate(train_set)
    if label in selected_classes
]


for i in selected_indices_train:
    train_set.targets[i]=label_mapping[train_set.targets[i].item()]

# 创建子集数据集
filtered_train_set = Subset(train_set, selected_indices_train)

train_loader = DataLoader(filtered_train_set, batch_size=64, shuffle=True, num_workers=4)



"""
# 加载  测试集
test_set = torchvision.datasets.MNIST(root='./data',
                               train=True,
                               download=False,
                               transform=transform)
# 筛选出所需类别的索引
selected_indices_test = [
    idx for idx, (_, label) in enumerate(test_set)
    if label in selected_classes
]


for i in selected_indices_test:
    test_set.targets[i]=label_mapping[test_set.targets[i].item()]

# 创建子集数据集
filtered_test_set = Subset(test_set, selected_indices_test)

test_loader = DataLoader(filtered_test_set, batch_size=64, shuffle=False, num_workers=4)
"""

print('Finish data loading')
print(f"Training data size: {len(filtered_train_set)}")
#print(f"Testing data size: {len(filtered_test_set)}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

model = Unet(
    dim = 64,
    channels = 1,
    dim_mults = (1, 2, 2)
)
model.to(device)
diffusion = GaussianDiffusion(
    model,
    objective = 'pred_noise',
    image_size = 28,
    timesteps = 500    # number of steps
)
diffusion.to(device)




def train(train_loader, epochs=10, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for i in range(epochs):
            
            loss_final = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            loop.set_description(f'Epoch [{i}/{epochs}]')
            for step, (images, labels) in loop:
    
                batch_size = images.shape[0]
                images = images.to(device)
                loss = diffusion(images)
                loss_final+=loss.item()

                
                
                optimizer.zero_grad()    
                loss.backward()
                optimizer.step()

                loop.set_postfix(loss=loss_final)
# torch.save(model.state_dict(),'test.pt')

train(train_loader,epochs=10,device=device)

save_dir = './MNIST/model'  # 目录
save_path = f'{save_dir}/MNIST_{se_index}.pt'  # 文件路径

# 确保目录存在，但不要创建文件
os.makedirs(save_dir, exist_ok=True)  # ✅ 只创建目录，不影响文件

# 保存模型
torch.save(model.state_dict(), save_path)  # ✅ 保存文件
print(f"Model saved at: {save_path}")
