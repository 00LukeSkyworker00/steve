import argparse, os
import numpy as np

import torch

from torch.utils.data import DataLoader

from steve import STEVE
from data import GlobVideoDatasetWithMasks
from ari import evaluate_ari


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=6)

parser.add_argument('--trained_model_paths', nargs='+', default=[
    'logs_for_seed_1/best_model_until_200000_steps.pt',
    'logs_for_seed_2/best_model_until_200000_steps.pt',
    'logs_for_seed_3/best_model_until_200000_steps.pt',
])
parser.add_argument('--data_path', default='eval_data/*')
parser.add_argument('--data_num_segs_per_frame', type=int, default=25)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

args = parser.parse_args()

torch.manual_seed(args.seed)

val_path = os.path.join(args.data_path, 'val', '*')
eval_dataset = GlobVideoDatasetWithMasks(
    root=val_path, phase='', img_size=args.image_size,
    ep_len=args.ep_len, img_glob='????????_image.png', mask_glob='????????_mask_??.png'
)
print(len(eval_dataset))
eval_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, **loader_kwargs)

models = []
for path in args.trained_model_paths:
    model = STEVE(args)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.cuda()
    models += [model]

with torch.no_grad():
    for model in models:
        model.eval()

    fgaris_img = []
    fgaris_vid = []
    for batch, (video, true_masks) in enumerate(eval_loader):
        video = video.cuda()

        fgaris_img_b = []
        fgaris_vid_b = []
        for model in models:
            _, _, pred_masks_b_m = model.encode(video)

            # omit the BG segment i.e. the 0-th segment from the true masks as follows.
            true_masks:torch.Tensor = true_masks[:, :, 1:]  # (B, T, N_obj, C, H, W)

            # Image FG-ARI
            fgari = 100 * evaluate_ari(true_masks.flatten(0,1).flatten(2),
                                           pred_masks_b_m.flatten(0,1).flatten(2))
            fgaris_img_b += [fgari]

            # Video FG-ARI
            fgari = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2),
                                           pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
            fgaris_vid_b += [fgari]

        fgaris_img += [fgaris_img_b]
        fgaris_vid += [fgaris_vid_b]

        # print results
        fgaris_numpy_img = np.asarray(fgaris_img)
        fgaris_numpy_vid = np.asarray(fgaris_vid)
        print(f"Done batches {batch + 1}. Over {len(models)} seeds. Ep Len {args.ep_len}.")
        
        mean = fgaris_numpy_img.mean(axis=0).mean()
        stddev = fgaris_numpy_img.mean(axis=0).std()
        print(f"IMAGE FG-ARI MEAN = {mean:.3f} \t STD = {stddev:.3f} .")

        mean = fgaris_numpy_vid.mean(axis=0).mean()
        stddev = fgaris_numpy_vid.mean(axis=0).std()
        print(f"VIDEO FG-ARI MEAN = {mean:.3f} \t STD = {stddev:.3f} .")
