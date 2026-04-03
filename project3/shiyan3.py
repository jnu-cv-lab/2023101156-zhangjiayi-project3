import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import os
import tkinter as tk
from tkinter import filedialog

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def select_image_file():
    """弹出文件选择对话框，选择图像文件"""
    root = tk.Tk()
    root.withdraw()
    file_types = [('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'), ('所有文件', '*.*')]
    file_path = filedialog.askopenfilename(title='选择图像文件', filetypes=file_types, initialdir=os.path.expanduser('~'))
    root.destroy()
    return file_path

def compute_mse_psnr(img_orig, img_rec):
    """计算两幅图像之间的 MSE 和 PSNR（dB）"""
    mse = np.mean((img_orig.astype(np.float64) - img_rec.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = 255.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return mse, psnr

def show_images(imgs, titles, figsize=(12, 8), cmap='gray'):
    """辅助显示多张图像"""
    n = len(imgs)
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def fft_spectrum(img):
    """计算图像的二维FFT频谱（中心化，对数幅度），返回频谱图像"""
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log(1 + magnitude)
    norm_log = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return norm_log.astype(np.uint8)

def dct_8x8_block(img, keep_low=6):
    """
    对图像进行8x8分块DCT，每块只保留keep_low个最低频系数（左上角）
    返回：重建图像、DCT系数图（用于显示）、低频能量占比
    """
    h, w = img.shape
    # 填充图像使其能被8整除
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    ph, pw = img_padded.shape
    
    # 复制图像用于处理
    img_f = img_padded.astype(np.float64)
    dct_coeff_full = np.zeros_like(img_f)
    
    # 分块DCT
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = img_f[i:i+8, j:j+8]
            # 对当前块做DCT
            dct_row = dct(block, type=2, norm='ortho', axis=0)
            dct_block = dct(dct_row, type=2, norm='ortho', axis=1)
            dct_coeff_full[i:i+8, j:j+8] = dct_block
    
    # 创建掩码：只保留左上角 keep_low 个系数
    # 使用Zigzag顺序的前keep_low个位置
    zigzag = [(0,0), (0,1), (1,0), (0,2), (1,1), (2,0), (0,3), (1,2), (2,1), (3,0),
              (0,4), (1,3), (2,2), (3,1), (4,0), (0,5), (1,4), (2,3), (3,2), (4,1),
              (5,0), (0,6), (1,5), (2,4), (3,3), (4,2), (5,1), (6,0), (0,7), (1,6),
              (2,5), (3,4), (4,3), (5,2), (6,1), (7,0), (1,7), (2,6), (3,5), (4,4),
              (5,3), (6,2), (7,1), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (3,7),
              (4,6), (5,5), (6,4), (7,3), (4,7), (5,6), (6,5), (7,4), (5,7), (6,6),
              (7,5), (6,7), (7,6), (7,7)]
    
    mask = np.zeros((8, 8))
    for idx in range(keep_low):
        if idx < len(zigzag):
            r, c = zigzag[idx]
            mask[r, c] = 1
    
    # 计算总能量和保留能量
    total_energy = 0
    kept_energy = 0
    
    # 重建图像
    img_reconstructed = np.zeros_like(img_f)
    
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            dct_block = dct_coeff_full[i:i+8, j:j+8]
            
            # 计算能量
            total_energy += np.sum(dct_block ** 2)
            kept_energy += np.sum((dct_block * mask) ** 2)
            
            # 应用掩码
            dct_block_masked = dct_block * mask
            # 逆DCT
            idct_row = idct(dct_block_masked, type=2, norm='ortho', axis=1)
            idct_block = idct(idct_row, type=2, norm='ortho', axis=0)
            img_reconstructed[i:i+8, j:j+8] = idct_block
    
    # 计算低频能量占比
    low_ratio = kept_energy / total_energy if total_energy > 0 else 0
    
    # 裁剪回原始尺寸
    img_reconstructed = np.clip(img_reconstructed[:h, :w], 0, 255).astype(np.uint8)
    
    # 生成DCT系数显示图（取对数，归一化）
    dct_log = np.log(1 + np.abs(dct_coeff_full))
    dct_log_norm = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 只显示原始尺寸部分
    dct_log_norm = dct_log_norm[:h, :w]
    
    return img_reconstructed, dct_log_norm, low_ratio

def downsample_and_recover(img, method='nearest'):
    """下采样并恢复图像"""
    h, w = img.shape
    small_w, small_h = w // 2, h // 2
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    if method == 'nearest':
        interp = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif method == 'cubic':
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR
    
    img_recovered = cv2.resize(img_small, (w, h), interpolation=interp)
    return img_small, img_recovered

def main():
    # 1. 弹窗选择图像文件
    print("请在弹出的对话框中选择图像文件...")
    img_path = select_image_file()
    
    if not img_path:
        print("未选择任何文件，程序退出。")
        return
    
    print(f"已选择图像: {img_path}")
    
    # 2. 读入灰度图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取图像文件 {img_path}")
        return
    
    h, w = img.shape
    print(f"原始图像尺寸: {h} x {w}")
    
    # 3. 下采样和恢复
    img_small_nn, img_rec_nn = downsample_and_recover(img, 'nearest')
    img_small_bil, img_rec_bil = downsample_and_recover(img, 'bilinear')
    img_small_cub, img_rec_cub = downsample_and_recover(img, 'cubic')
    img_small = img_small_nn
    
    # 4. 空间域比较
    print("\n===== 空间域评价 =====")
    show_images([img, img_small, img_rec_nn, img_rec_bil, img_rec_cub],
               ['Original Image', f'Downsampled (1/2)', 'Nearest Neighbor', 'Bilinear', 'Cubic'],
                figsize=(15, 4))
    
    for name, rec_img in zip(['最近邻', '双线性', '三次内插'], [img_rec_nn, img_rec_bil, img_rec_cub]):
        mse, psnr = compute_mse_psnr(img, rec_img)
        print(f"{name:8s}: MSE = {mse:.4f}, PSNR = {psnr:.2f} dB")
    
    # 5. 傅里叶变换分析
    print("\n===== 傅里叶频谱分析 =====")
    spec_orig = fft_spectrum(img)
    spec_small = fft_spectrum(img_small)
    spec_rec = fft_spectrum(img_rec_bil)
    
    show_images([spec_orig, spec_small, spec_rec],
                ['Original Spectrum', 'Downsampled Spectrum', 'Bilinear Reconstructed Spectrum'],
                figsize=(12, 4))
    
    # 6. DCT 分析（8x8分块，每块只保留6个最低频系数）
    print("\n===== DCT 分析 (8x8分块，每块保留6/64系数) =====")
    
    # 对原图进行8x8 DCT分块处理
    img_dct_recon, dct_coeff_img, low_ratio_orig = dct_8x8_block(img, keep_low=6)
    
    # 对三种恢复图像进行相同的DCT处理
    img_dct_nn, dct_coeff_nn, low_ratio_nn = dct_8x8_block(img_rec_nn, keep_low=6)
    img_dct_bil, dct_coeff_bil, low_ratio_bil = dct_8x8_block(img_rec_bil, keep_low=6)
    img_dct_cub, dct_coeff_cub, low_ratio_cub = dct_8x8_block(img_rec_cub, keep_low=6)
    
    # 显示DCT系数图（对数显示）
    plt.figure(figsize=(16, 10))
   
    #频谱图中心最亮，对应人像的平滑轮廓等低频信息；四周较暗，代表边缘、纹理等高频细节。
    plt.subplot(2, 4, 1)
    plt.imshow(dct_coeff_nn, cmap='gray')
    plt.title(f'Nearest Neighbor Reconstructed DCT Coefficients')
    #十字亮线源于图像中常见的水平和垂直结构（如肩膀、发际线）。
    plt.subplot(2, 4, 2)
    plt.imshow(dct_coeff_bil, cmap='gray')
    plt.title(f'Bilinear Reconstructed DCT Coefficients')
    #下采样后高频丢失或混叠，频谱四周变暗；双线性恢复进一步抑制高频，使能量更向中心集中，说明丢失的细节无法复原。
    plt.subplot(2, 4, 3)
    plt.imshow(dct_coeff_cub, cmap='gray')
    plt.title(f'Cubic Reconstructed DCT Coefficients')
    
    plt.subplot(2, 4, 5)
    plt.imshow(img_dct_nn, cmap='gray')
    plt.title(f'Nearest Neighbor Reconstructed DCT\nEnergy Retention: {low_ratio_nn*100:.1f}%')
    
    plt.subplot(2, 4, 6)
    plt.imshow(img_dct_bil, cmap='gray')
    plt.title(f'Bilinear Reconstructed DCT\nEnergy Retention: {low_ratio_bil*100:.1f}%')
    
    plt.subplot(2, 4, 7)
    plt.imshow(img_dct_cub, cmap='gray')
    plt.title(f'Cubic Reconstructed DCT\nEnergy Retention: {low_ratio_cub*100:.1f}%')
    
    plt.subplot(2, 4, 8)
    plt.imshow(img_dct_recon, cmap='gray')
    plt.title(f'8x8 DCT Block Reconstruction (6 coefficients per block)\nEnergy Retention: {low_ratio_orig*100:.1f}%')
    
    plt.subplot(2, 4, 4)
    diff = cv2.absdiff(img, img_dct_recon)
    plt.imshow(diff, cmap='gray')
    plt.title('Reconstruction Error Map (Brighter = Larger Error)')
    
    plt.tight_layout()
    plt.show()
    
    # 打印DCT保留能量占比
    print("\n【8x8 DCT分块保留能量占比 (每块保留6/64系数)】")
    print("-" * 50)
    print(f"{'图像类型':<12} | {'保留能量占比':<15}")
    print("-" * 50)
    print(f"{'Original Image':<12} | {low_ratio_orig*100:>14.2f}%")
    print(f"{'Nearest Neighbor':<12} | {low_ratio_nn*100:>14.2f}%")
    print(f"{'Bilinear':<12} | {low_ratio_bil*100:>14.2f}%")
    print(f"{'Cubic':<12} | {low_ratio_cub*100:>14.2f}%")
    print("-" * 50)
    
    plt.show()
    print("\n【高频成分差异分析】")
    print("1. 原图频谱：包含完整的高低频信息，人像边缘对应高频分量。")
    print("2. 缩小图频谱：高频成分发生混叠，频谱图中可见能量向中心折叠。")
    print("3. 双线性恢复图频谱：双线性插值相当于低通滤波，高频成分被抑制，图像变模糊。")
    print("\n【DCT分析结论】")
    print("1. 人像的DCT系数能量主要集中在低频区域（左上角），这正是图像压缩的基础。")
    print("2. 仅保留1-5%的DCT系数仍能识别人像，体现了DCT的能量集中特性。")
    print("3. 不同恢复方法的DCT能量分布差异：")
    print("   - 最近邻恢复：产生高频伪影，低频能量占比最低")
    print("   - 双线性恢复：平滑效果明显，低频能量占比最接近原图")
    print("   - 三次内插恢复：介于两者之间，能保留更多边缘信息")
    print("4. 从低频重建图可以看出，人像的主要轮廓和明暗关系由低频成分决定。")
    print("\n处理完成！")

if __name__ == '__main__':
    main()