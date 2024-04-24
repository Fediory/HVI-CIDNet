import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
from os.path import join
from os import listdir
import argparse

mea_parser = argparse.ArgumentParser(description='Measure')
mea_parser.add_argument('--use_GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
mea_parser.add_argument('--SID', action='store_true')
mea_parser.add_argument('--Blur', action='store_true')
mea = mea_parser.parse_args()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def metrics(im_dir, label_dir, use_GT_mean):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    

    for item in sorted(glob.glob(im_dir)):
        n += 1
        im1 = Image.open(item).convert('RGB') 

        name = item.split('\\')[-1]
        
        if mea.SID:
            data_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
            im2 = Image.open(data_filenames[0]).convert('RGB')
        else:
            im2 = Image.open(label_dir + name).convert('RGB')
            
        (h, w) = im2.size
        im1 = im1.resize((h, w))  
        im1 = np.array(im1)
        im2 = np.array(im2)
        
        if use_GT_mean:
            mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
            im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)
            
        score_psnr = calculate_psnr(im1, im2)
        score_ssim = calculate_ssim(im1, im2)

        ex_p0 = lpips.im2tensor(im1)
        ex_ref = lpips.im2tensor(im2)
        ex_p0 = ex_p0.cuda()
        ex_ref = ex_ref.cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0)
    
        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips
    return avg_psnr, avg_ssim, avg_lpips, n


if __name__ == '__main__':
    
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    if mea.Blur:
        for index in range(1,257):
            fill_index = str(index).zfill(4)
            test_dir = "./output/LOL_Blur/"
            im_dir = test_dir + fill_index + "/*.png"
            label_dir = './datasets/LOL_blur/test/high_sharp_scaled/' + fill_index + "/"
            if os.path.exists(test_dir + fill_index):
                i_psnr, i_ssim, i_lpips, i_n = metrics(im_dir, label_dir,mea.use_GT_mean)
                print("===> Finish " + fill_index + " folder")
                print("===> Avg.PSNR: {:.4f} dB ".format(i_psnr/i_n))
                print("===> Avg.SSIM: {:.4f} ".format(i_ssim/i_n))
                print("===> Avg.LPIPS: {:.4f}\n ".format(i_lpips.item()/i_n))
                avg_psnr    += i_psnr
                avg_ssim    += i_ssim
                avg_lpips   += i_lpips.item()
                n += i_n
                torch.cuda.empty_cache()
    
    elif mea.SID:
        for index in range(1,257):
            fill_index = "1"+str(index).zfill(4)
            test_dir = "./output/SID/"
            im_dir = test_dir + fill_index + "/*.png"
            label_dir = './datasets/Sony_total_dark/test/long/' + fill_index + "/"
            if os.path.exists(test_dir + fill_index):
                i_psnr, i_ssim, i_lpips, i_n = metrics(im_dir, label_dir,mea.use_GT_mean)
                print("===> Finish " + fill_index + " folder")
                print("===> Avg.PSNR: {:.4f} dB ".format(i_psnr/i_n))
                print("===> Avg.SSIM: {:.4f} ".format(i_ssim/i_n))
                print("===> Avg.LPIPS: {:.4f}\n ".format(i_lpips.item()/i_n))
                avg_psnr    += i_psnr
                avg_ssim    += i_ssim
                avg_lpips   += i_lpips.item()
                n += i_n
                torch.cuda.empty_cache()
        
    print("===> All Finish")
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr/n))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim/n))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips/n))
