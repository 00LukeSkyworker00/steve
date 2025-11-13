import os
import argparse
import tensorflow_datasets as tfds
import torchvision.utils as vutils
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', default='data/')
parser.add_argument('--level', default='e')
parser.add_argument('--split', default='train')
parser.add_argument('--version', default='1.0.0')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--local_data_dir', default='~/tensorflow_datasets', help='Local TFDS cache path')
args = parser.parse_args()

dataset_name = f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}"
local_data_dir = os.path.expanduser(args.local_data_dir)

print(f"📦 Downloading (or loading cached) dataset: {dataset_name}")
ds, ds_info = tfds.load(
    dataset_name,
    data_dir=local_data_dir,
    with_info=True,
    try_gcs=False  # ✅ Force HTTPS/local mirror, no GCS dependency
)
print(f"✅ Dataset ready at {local_data_dir}")

train_iter = iter(tfds.as_numpy(ds[args.split]))
to_tensor = transforms.ToTensor()

for b, record in enumerate(train_iter):
    video = record['video']
    T, *_ = video.shape

    path_vid = os.path.join(args.out_path, f"{b:08}")
    os.makedirs(path_vid, exist_ok=True)

    for t in range(T):
        img = to_tensor(video[t])
        vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))

    print(f"✅ Saved video {b} to {path_vid}")
