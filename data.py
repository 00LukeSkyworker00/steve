import os
import glob
import torch

import numpy as np

from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, img_size, ep_len=3, img_glob='*.png'):
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video


class GlobVideoDatasetWithMasks(Dataset):
    def __init__(self, root, phase, img_size, ep_len=6, img_glob='*_image.png', mask_glob='*_mask_*.png'):
        self.root = root
        self.img_size = img_size
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

        if phase == 'train':
            self.total_dirs = self.total_dirs[:int(len(self.total_dirs) * 0.7)]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.7):int(len(self.total_dirs) * 0.85)]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[int(len(self.total_dirs) * 0.85):]
        else:
            pass

        # chunk into episodes
        self.episodes_rgb = []
        self.episodes_mask = []
        for dir in self.total_dirs:
            frame_buffer = []
            mask_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            mask_paths = sorted(glob.glob(os.path.join(dir, mask_glob)))
            avg_num_mask = len(mask_paths) // len(image_paths)
            base_idx = 0
            for image_path in image_paths:
                p = Path(image_path)
                frame_buffer.append(p)

                if avg_num_mask == 1:
                    masks = Path(mask_paths[avg_num_mask])
                else:
                    masks=[]
                    for i in range(base_idx, base_idx + avg_num_mask):
                        p = Path(mask_paths[i])
                        masks.append(p)

                base_idx += avg_num_mask
                mask_buffer.append(masks)
                
                if len(frame_buffer) == self.ep_len:
                    self.episodes_rgb.append(frame_buffer)
                    self.episodes_mask.append(mask_buffer)
                    frame_buffer = []
                    mask_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes_rgb)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes_rgb[idx]:
            image = Image.open(img_loc).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)

        masks = []
        for mask_locs in self.episodes_mask[idx]:
            frame_masks = []
            if isinstance(mask_locs, list):
                for mask_loc in mask_locs:
                    image = Image.open(mask_loc).convert('1')
                    image = image.resize((self.img_size, self.img_size))
                    tensor_image = self.transform(image)
                    frame_masks += [tensor_image]
                frame_masks = torch.stack(frame_masks, dim=0)
                masks += [frame_masks]
            else:
                image = Image.open(mask_loc).convert('L')
                image = image.resize((self.img_size, self.img_size))
                tensor_image = self.transform(image)
                one_hot = F.one_hot(tensor_image).permute(2,0,1).bool()
                masks += [one_hot]
        masks = torch.stack(masks, dim=0)

        return video, masks

class FlowDataset(Dataset):
    def __init__(self, root, load_mask=False):
        self.root = root
        self.total_dirs = sorted(glob.glob(os.path.join(root, "*")))
        self.load_mask = load_mask

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_dirs)

    def __getitem__(self, idx):
        flow_path = os.path.join(self.total_dirs[idx], 'fwd.npy')
        flow = np.load(flow_path, mmap_mode="r")
        flow = torch.from_numpy(flow)  # (F, H, W, 2)
        zeros = torch.zeros_like(flow[...,:1])
        flow_vid = torch.cat([flow,zeros], dim=-1)  # (F, H, W, 3)
        flow_vid = flow_vid.permute(0,3,1,2)  # (F, 3, H, W)
        flow_vid = flow_vid[:21]  # (21, 3, H, W), drop nan in the end

        if self.load_mask:
            mask_path = os.path.join(self.total_dirs[idx], 'mask_gt.npy')
            masks = np.load(mask_path, mmap_mode="r")
            masks = torch.from_numpy(masks).bool().float()  # (F, obj, H, W, 1)
            masks = masks.permute(0,1,4,2,3)[:21]  # (21, obj, 1, H, W), drop nan in the end
            return flow_vid, masks
        else:
            return flow_vid