from pytorch_fid import fid_score

def compute_fid(path_real, path_fake, batch_size=50, device="cuda", dims=2048):
    """
    计算两个路径下的 FID 分数
    - path_real: 真实图片的文件夹路径
    - path_fake: 生成图片的文件夹路径
    - batch_size: 计算 FID 时的 batch_size（默认 50）
    - device: 计算 FID 的设备（默认 "cuda"，可选 "cpu"）
    - dims: 使用 InceptionV3 的哪个层（默认 2048）
    
    返回:
    - FID 分数（数值，越低越好）
    """
    fid_value = fid_score.calculate_fid_given_paths(
        paths=[path_real, path_fake],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    
    print(f"✅ FID Score: {fid_value:.4f}")
    return fid_value
