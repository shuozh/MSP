import os
import torch
import time
import imageio
import math
import argparse
from utils import llf_dataset
import numpy as np
import torch.nn as nn
from models import LFEn_s3
from torch.utils.data import DataLoader
import lib.pytorch_ssim as pytorch_ssim
import torchvision
import lpips

from skimage.metrics import structural_similarity as _SSIM
from skimage.metrics import peak_signal_noise_ratio as _PSNR

# from skimage.measure import compare_ssim as _SSIM
# from skimage.measure import compare_psnr as _PSNR

torch.set_num_threads(8)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--data_dir", type=str, default="../../../datasets/LowLightLF")
    parser.add_argument('-o', "--output_dir", type=str, default="./")
    parser.add_argument('-exp', "--exp_name", type=str, default="")
    parser.add_argument('-d', "--dataset", type=str, default="1_100")

    parser.add_argument('-e', "--epochs", type=int, default=10000)
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-4)
    parser.add_argument('-gpu', "--gpu_no", type=int, default=0)
    parser.add_argument('-p', "--patch", type=int, default=128)
    parser.add_argument('-n', "--n_view", type=int, default=5)
    parser.add_argument('-pai', "--is_pai", type=bool, default=False)

    return parser.parse_args()

def overlap_crop_forward(x, hist_data, model, device, overlap=20, patch_size=256):
    N, an2, c, h, w = x.shape

    output = torch.zeros(N, an2, c, h, w).to(device)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            s1, e1 = i, (i + patch_size) if (i + patch_size) < h else h
            s2, e2 = j, (j + patch_size) if (j + patch_size) < w else w
            #
            s11, e11 = s1 - overlap if s1 != 0 else 0, e1 + overlap
            s22, e22 = s2 - overlap if s2 != 0 else 0, e2 + overlap

            patch_input = x[:, :, :, s11:e11, s22:e22]
            patch_input = patch_input.to(device)

            outs = model(patch_input)
            patch_out = outs[-1]

            s1_out = overlap if s1 != 0 else 0
            s2_out = overlap if s2 != 0 else 0

            output[:, :, :, s1:e1, s2:e2] = patch_out[:, :, :, s1_out:s1_out + patch_size,
                                            s2_out:s2_out + patch_size]

    return output


def test_20(args):
    device = torch.device("cuda:{}".format(args.gpu_no))
    model = LFEn_s3(args.n_view).to(device)
  
    args.dataset = "1_20"
    
    model.load_state_dict(torch.load('weights/1_20/weights.pkl'))
    model.eval()
    count = 0

    ssim = pytorch_ssim.SSIM(val_range=1)
    ssim.to(device)
    l2 = nn.MSELoss()

    psnr_list, ssim_list = [], []
    psnr_list1, ssim_list1 = [], []
    lpips_list = []

    data_test = llf_dataset(args, is_train=False)
    # batch_size for testing must be 1
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    t1 = time.time()

    lpips_func = lpips.LPIPS(net='alex')

    with torch.no_grad():
        for idx, (input, gt) in enumerate(dataloader_test):
            data, gt = input.to(device), gt.to(device)
            # hist = hist.to(device)
            N, an2, c, h, w = gt.shape
            an = args.n_view
            t3 = time.time()
            out = overlap_crop_forward(input, None, model, device, overlap=30, patch_size=256)
            t4 = time.time()

            out = torch.clamp(out, 0, 1)

            out_saved = out.reshape(N,  5, 5, c, h, w).permute(0, 1, 2, 4, 5, 3).cpu().numpy()
            gt_saved = gt.reshape(N, 5, 5, c, h, w).permute(0, 1, 2, 4, 5, 3).cpu().numpy()


            for i in range(N):
                view_psnr = []
                view_ssim = []
                for ii in range(an2):
                    mse = l2(gt[i, ii].unsqueeze(0), out[i, ii].unsqueeze(0))
                    psnr = 10 * np.log10(1 / mse.item())
                    view_psnr.append(psnr)
                    ssim_ = ssim(out[i, ii].unsqueeze(0), gt[i, ii].unsqueeze(0)).item()
                    view_ssim.append(ssim_)
                psnr_list1.append(np.mean(view_psnr))
                ssim_list1.append(np.mean(view_ssim))

            out = out.reshape(N * an2, c, h, w)
            gt = gt.reshape(N * an2, c, h, w)

            #for lpips, we calculate the center view only
            out_cal = out.cpu()[an2 // 2, :, :, :].cpu()
            gt_cal = gt.cpu()[an2 // 2, :, :, :].cpu()
            lpips_metric = lpips_func(out_cal, gt_cal)
            lpips_list.append(lpips_metric.item())

            print('==>IMG_{} PSNR = {:.2f}, SSIM = {:.2f}, LPIPS = {:.4f}'.format(idx + 1, psnr_list1[-1], ssim_list1[-1], lpips_metric.item()))
            count += 1

    t2 = time.time()
    print('===>Test Average PSNR = {:.2f}, SSIM: {:.2f},LPIPS = {:.4f} Time: {:.2f}s'.format(np.mean(psnr_list1), np.mean(ssim_list1), np.mean(lpips_list), t2 - t1))
    torch.cuda.empty_cache()


def test_50(args):
    device = torch.device("cuda:{}".format(args.gpu_no))
    model = LFEn_s3(args.n_view).to(device)
  
    args.dataset = "1_50"

    model.load_state_dict(torch.load('weights/1_50/weights.pkl'))
    model.eval()
    count = 0

    ssim = pytorch_ssim.SSIM(val_range=1)
    ssim.to(device)
    l2 = nn.MSELoss()

    psnr_list, ssim_list = [], []
    psnr_list1, ssim_list1 = [], []
    lpips_list = []

    data_test = llf_dataset(args, is_train=False)
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    t1 = time.time()

    lpips_func = lpips.LPIPS(net='alex')

    with torch.no_grad():
        for idx, (input, gt) in enumerate(dataloader_test):
            data, gt = input.to(device), gt.to(device)
            N, an2, c, h, w = gt.shape

            an = args.n_view

            t3 = time.time()
            out = overlap_crop_forward(input, None, model, device, overlap=30, patch_size=256)

            t4 = time.time()
            out = torch.clamp(out, 0, 1)

            out_saved = out.reshape(N,  5, 5, c, h, w).permute(0, 1, 2, 4, 5, 3).cpu().numpy()
            gt_saved = gt.reshape(N, 5, 5, c, h, w).permute(0, 1, 2, 4, 5, 3).cpu().numpy()

            for i in range(N):
                view_psnr = []
                view_ssim = []
                for ii in range(an2):
                    mse = l2(gt[i, ii].unsqueeze(0), out[i, ii].unsqueeze(0))
                    psnr = 10 * np.log10(1 / mse.item())
                    view_psnr.append(psnr)
                    ssim_ = ssim(out[i, ii].unsqueeze(0), gt[i, ii].unsqueeze(0)).item()
                    view_ssim.append(ssim_)
                    # view_psnr.append(_PSNR(out[i, ii].permute(1, 2, 0).cpu().numpy(), gt[i, ii].permute(1, 2, 0).cpu().numpy()))
                psnr_list1.append(np.mean(view_psnr))
                ssim_list1.append(np.mean(view_ssim))

          
            out = out.reshape(N * an2, c, h, w)
            gt = gt.reshape(N * an2, c, h, w)
        
            #for lpips, we calculate the center view only
            out_cal = out.cpu()[an2 // 2, :, :, :].cpu()
            gt_cal = gt.cpu()[an2 // 2, :, :, :].cpu()
            lpips_metric = lpips_func(out_cal, gt_cal)
            lpips_list.append(lpips_metric.item())

            print('==>IMG_{} PSNR = {:.2f}, SSIM = {:.2f}, LPIPS = {:.4f}'.format(idx + 1, psnr_list1[-1], ssim_list1[-1], lpips_metric.item()))
            count += 1

    t2 = time.time()

    print('===>Test Average PSNR = {:.2f}, SSIM: {:.2f},LPIPS = {:.4f} Time: {:.2f}s'.format(np.mean(psnr_list1), np.mean(ssim_list1), np.mean(lpips_list), t2 - t1))
    torch.cuda.empty_cache()


def test_100(args):
    device = torch.device("cuda:{}".format(args.gpu_no))
    model = LFEn_s3(args.n_view).to(device)

    args.dataset = "1_100"
    
    model.load_state_dict(torch.load('weights/1_100/weights.pkl'))
    model.eval()
    count = 0

    ssim = pytorch_ssim.SSIM(val_range=1)
    ssim.to(device)
    l2 = nn.MSELoss()

    psnr_list, ssim_list = [], []
    psnr_list1, ssim_list1 = [], []
    lpips_list = []

    data_test = llf_dataset(args, is_train=False)
    # batch_size for testing must be 1
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    t1 = time.time()

    lpips_func = lpips.LPIPS(net='alex')

    with torch.no_grad():
        for idx, (input, gt) in enumerate(dataloader_test):
            data, gt = input.to(device), gt.to(device)
            N, an2, c, h, w = gt.shape

            an = args.n_view

            t3 = time.time()
            out = overlap_crop_forward(input, None, model, device, overlap=30, patch_size=256)
            t4 = time.time()

            out = torch.clamp(out, 0, 1)
           
            for i in range(N):
                view_psnr = []
                view_ssim = []
                view_psnr1 = []
                view_ssim1 = []
                for ii in range(an2):
                    mse = l2(gt[i, ii].unsqueeze(0), out[i, ii].unsqueeze(0))
                    psnr = 10 * np.log10(1 / mse.item())
                    view_psnr.append(psnr)
                    ssim_ = ssim(out[i, ii].unsqueeze(0), gt[i, ii].unsqueeze(0)).item()
                    view_ssim.append(ssim_)
                psnr_list1.append(np.mean(view_psnr))
                ssim_list1.append(np.mean(view_ssim))
          
            out = out.reshape(N * an2, c, h, w)
            gt = gt.reshape(N * an2, c, h, w)

            out_cal = out.cpu()[an2 // 2, :, :, :].cpu()
            gt_cal = gt.cpu()[an2 // 2, :, :, :].cpu()
            lpips_metric = lpips_func(out_cal, gt_cal)
            lpips_list.append(lpips_metric.item())

        

            print('==>IMG_{} PSNR = {:.2f}, SSIM = {:.2f}, LPIPS = {:.4f}'.format(idx + 1, psnr_list1[-1], ssim_list1[-1], lpips_metric.item()))
            count += 1

    t2 = time.time()
    print('===>Test Average PSNR = {:.2f}, SSIM: {:.2f},LPIPS = {:.4f} Time: {:.2f}s'.format(np.mean(psnr_list1), np.mean(ssim_list1), np.mean(lpips_list), t2 - t1))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    
    test_100(args)
    test_50(args)
    test_20(args)
    
