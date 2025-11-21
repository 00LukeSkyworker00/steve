import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from steve import STEVE
from data import GlobVideoDataset, GlobVideoDatasetWithMasks
from utils import cosine_anneal, linear_warmup
from ari import evaluate_ari

import matplotlib.pyplot as plt
import wandb

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/*')
parser.add_argument('--out_path', default='results/')
parser.add_argument('--dataset', default='movi_solid')
parser.add_argument('--log_samples', type=int, default=4)

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=200000)

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

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=True, action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

ds_type = args.dataset
args_dict = vars(args)
out_dir = os.path.join(args.out_path, datetime.today().isoformat())
os.makedirs(out_dir, exist_ok=True)

# Setup WanDB
wandb.login()
run = wandb.init(
    project=f'steve_pretrain_{ds_type}',
    config=args_dict,
    name=datetime.today().strftime("%y%m%d_%H%M%S")
)
# Define custom step axes:
run.define_metric("TRAIN/*", step_metric="train_step", hidden=True)
run.define_metric("VAL/*", step_metric="val_step", hidden=True)

# Setup Tensorboard
arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(out_dir, 'tensorboard')
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

train_path = os.path.join(args.data_path, 'train', '*')
val_path = os.path.join(args.data_path, 'val', '*')

if ds_type in ('movi-flow'):
    img_glob = 'movi_?_????_rgb_???.jpg'
    mask_glob = 'movi_?_????_ano_???.jpg'
else:
    img_glob = '????????_image.png'
    mask_glob = '????????_mask_??.png'

train_dataset = GlobVideoDataset(
    root=train_path, phase='', 
    img_size=args.image_size, ep_len=args.ep_len, 
    img_glob=img_glob)

val_dataset = GlobVideoDatasetWithMasks(
    root=val_path, img_size=args.image_size,
    ep_len=args.ep_len, img_glob=img_glob, mask_glob=mask_glob
)

loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=None, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=None, shuffle=False, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

val_sample = val_dataset[0]
gt_num_slot = val_sample[1].shape[1]

log_interval = max(train_epoch_size // 5, 1)
log_samples = 4
seg_cmap = plt.cm.tab20(torch.linspace(0, 1, gt_num_slot))[:, :3]
seg_cmap = torch.from_numpy(seg_cmap).cuda()
seg_cmap[0] *= 0

model = STEVE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

def visualize(video:torch.Tensor, recon_dvae:torch.Tensor, recon_tf:torch.Tensor, 
              seg_gt:torch.Tensor, seg_pred:torch.Tensor, attns:torch.Tensor, N=8):
    _, T, C, H, W = video.size()

    video = video[:N].unsqueeze(2)  # (N, T, 1, C, H, W)
    recon_dvae = recon_dvae[:N].unsqueeze(2)  # (N, T, 1, C, H, W)
    recon_tf = recon_tf[:N].unsqueeze(2)  # (N, T, 1, C, H, W)
    seg_gt = seg_gt[:N].unsqueeze(2)  # (N, T, 1, C, H, W)
    seg_pred = seg_pred[:N].unsqueeze(2)  # (N, T, 1, C, H, W)
    attns = attns[:N]  # (N, T, S, C, H, W)

    num_slot_modulo = attns.shape[2] % 5
    if num_slot_modulo != 0:
        empty_attns = torch.ones_like(attns[:,:,:1].repeat(1,1,5-num_slot_modulo,1,1,1))
        attns = torch.cat([attns, empty_attns*0.5], dim=2)

    tiles = torch.cat((video, recon_dvae, recon_tf, seg_gt, seg_pred, attns), dim=2)  # (N, T, (S+5), C, H, W)
    tiles = tiles.permute(0,2,1,3,4,5).reshape(-1,T*C,H,W)  # (N*(S+5), T*C, H, W)
    
    frames = vutils.make_grid(tiles, nrow=(5), pad_value=0.8)  # (T*C, H', W')
    frames = frames.reshape(T, C, frames.shape[-2], frames.shape[-1]).unsqueeze(0)  # (1, T*C, H', W')

    return frames

stop_signal = False
for epoch in range(start_epoch, args.epochs):
    model.train()
    global_step = epoch * train_epoch_size

    if stop_signal:
        break

    for batch, video in enumerate(train_loader):
        global_step += 1
        if global_step > args.steps:
            stop_signal = True
            break

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        video = video.cuda()

        optimizer.zero_grad()
        
        (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)

        if args.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = mse + cross_entropy
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        
        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                run.log({
                    "TRAIN/loss": loss.item(),
                    "TRAIN/cross_entropy": cross_entropy.item(),
                    "TRAIN/mse": mse.item(),
                    "TRAIN/tau": tau,
                    "TRAIN/lr_dvae": optimizer.param_groups[0]['lr'],
                    "TRAIN/lr_enc": optimizer.param_groups[1]['lr'],
                    "TRAIN/lr_dec": optimizer.param_groups[2]['lr'],
                    "train_step": global_step,
                })
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)


    
    with torch.no_grad():
        model.eval()

        val_cross_entropy = 0.
        val_mse = 0.
        aris = 0.
        fgaris = 0.

        for batch, (video, true_masks) in enumerate(val_loader):
            video = video.cuda()

            (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)
            _, _, pred_masks = model.module.encode(video)

            #compute ari
            ari_b = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2),
                                        pred_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
            aris += ari_b

            # compute fg-ari by omit the BG segment i.e. the 0-th segment from the true masks as follows.
            fgari_b = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                                        pred_masks.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
            fgaris += fgari_b

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()

        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        
        aris /= (val_epoch_size)
        fgaris /= (val_epoch_size)

        val_loss = val_mse + val_cross_entropy
            
        def viz_seg(attns: torch.Tensor):
            """                
            Args:
                attns (torch.Tensor): Segmentation map shaped (b,t,s,1,h,w)
            """
            obj_ids = torch.argmax(attns[:,:,:,0], dim=2)  # (b,t,h,w)
            colored_seg = seg_cmap[obj_ids]  # (b,t,h,w,3)
            return colored_seg.permute(0,1,4,2,3)
            
        gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[::8][:log_samples])
        seg_gt = viz_seg(true_masks[::8][:log_samples].cuda())
        seg_pred = viz_seg(pred_masks[::8][:log_samples])
        frames = visualize(video[::8], recon[::8], gen_video, seg_gt, seg_pred, attns[::8], N=log_samples)

        print('====> Epoch: {:3} \t Loss: {:F} \t Ari: {:F} \t FG-Ari: {:F}'.format(epoch+1, val_loss, aris, fgaris))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_pth =os.path.join(out_dir, 'best_model.pt')
            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), best_pth)

            # if global_step < args.steps:
            #     torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, f'best_model_until_{args.steps}_steps.pt'))

        run.log({
            "VAL/loss": val_loss,
            "VAL/cross_entropy": val_cross_entropy,
            "VAL/mse": val_mse,
            "VAL/ari": aris,
            "VAL/ari_fg": fgaris,
            "VAL/best_loss": best_val_loss,
            "VAL/recons": wandb.Video(frames.cpu()*255.0, fps=8, format="mp4"),
            "val_step": epoch+1,
        })

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch+1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)
        writer.add_scalar('VAL/ari', aris, epoch+1)
        writer.add_scalar('VAL/ari_fg', fgaris, epoch+1)
        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)
        writer.add_video('Recons/val', frames, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        ckpt_pth = os.path.join(out_dir, 'checkpoint.pt.tar')
        torch.save(checkpoint, ckpt_pth)


        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

artifact_best = wandb.Artifact("best", type="model")
artifact_best.add_file(best_pth)
artifact_best.ttl = None
run.log_artifact(artifact_best)

artifact_ckpt = wandb.Artifact("ckpt", type="checkpoint")
artifact_ckpt.add_file(ckpt_pth)
artifact_ckpt.ttl = None
run.log_artifact(artifact_ckpt)

artifact_log = wandb.Artifact("tensorboard", type="log")
artifact_log.add_dir(log_dir)
artifact_log.ttl = None
run.log_artifact(artifact_log)

writer.close()
run.finish()
