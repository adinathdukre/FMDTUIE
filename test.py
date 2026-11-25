"""Test script for 'Frequency Modulated Deformable Transformer for Underwater Image Enhancement' (ICPR 2024)."""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import network
from utils import UIEDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test 'Frequency Modulated Deformable Transformer for Underwater Image Enhancement'")
    parser.add_argument('--data_path_test', type=str, default='./datasets/images/', help='folder of test images')
    parser.add_argument('--data_name', type=str, default='UIEB', help='name of the output sub-folder')
    parser.add_argument('--mode', type=str, default='paired', choices=['paired', 'single'],
                        help="'paired': input|GT side by side in one file (reports PSNR/SSIM); "
                             "'single': plain underwater images")
    parser.add_argument('--size', type=int, default=512, help='square test resolution (divisible by 32)')
    parser.add_argument('--keep_size', action='store_true',
                        help='resize each output back to the resolution of its input')
    parser.add_argument('--save_path', type=str, default='result', help='outputs go to <save_path>/<data_name>/')
    parser.add_argument('--model_file', type=str, default='./checkpoint/UIEB.pth', help='pretrained checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

    # architecture (defaults match the released checkpoint)
    parser.add_argument('--inp_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 2])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--ffn_expansion_factor', type=int, default=3)
    return parser.parse_args()


def compute_metrics(out, gt):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnr = peak_signal_noise_ratio(gt, out, data_range=255)
    ssim = structural_similarity(gt, out, channel_axis=2, data_range=255)
    return psnr, ssim


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    model = network(args.inp_channels, args.out_channels, args.dim, args.num_blocks, args.num_refinement_blocks,
                    args.ffn_expansion_factor, bias=False).to(device)
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    dataset = UIEDataset(args.data_path_test, mode=args.mode, size=args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    out_dir = os.path.join(args.save_path, args.data_name)
    os.makedirs(out_dir, exist_ok=True)

    psnrs, ssims = [], []
    for inp, gt, name, h, w in tqdm(loader, desc='Enhancing'):
        out = model(inp.to(device)).clamp(0, 1)
        out = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        if args.mode == 'paired':
            gt_np = (gt[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            p, s = compute_metrics(out, gt_np)
            psnrs.append(p)
            ssims.append(s)

        img = Image.fromarray(out)
        if args.keep_size:
            img = img.resize((int(w), int(h)), Image.BICUBIC)
        img.save(os.path.join(out_dir, name[0]))

    if psnrs:
        print(f'PSNR: {np.mean(psnrs):.2f} dB | SSIM: {np.mean(ssims):.4f} over {len(psnrs)} images')
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
