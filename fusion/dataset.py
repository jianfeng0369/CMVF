import os
import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd

class FusionDatasetSeq(Dataset):
    def __init__(self, vi_path, ir_path, flow_path, seq_path, batch_size, transform=None):
        self.transform = transform
        self.vi_image_all_path = sorted(glob.glob(os.path.join(vi_path, '*.jpg')) + glob.glob(os.path.join(vi_path, '*.png')) + glob.glob(os.path.join(vi_path, '*.bmp')))

        self.vi_path = vi_path
        self.ir_path = ir_path
        self.flow_path = flow_path

        self.seq_path = seq_path
        self.batch_size = batch_size

        self.sequences = self.create_sequences()

        print(f"原有总图像 {len(self.vi_image_all_path)}")
        print(f"现有 Sequence {len(self.sequences)} * batch_size {self.batch_size} = {len(self.sequences) * self.batch_size}")

    def create_sequences(self):
        sequences = []
        data = pd.read_excel(self.seq_path)

        for _, row in data.iterrows():
            start_val = row.iloc[0]
            end_val   = row.iloc[1]


            def is_int_like(x):
                try:
                    int(x)
                    return "_" not in str(x)
                except (ValueError, TypeError):
                    return False

            if is_int_like(start_val) and is_int_like(end_val):

                start_idx = int(start_val)
                end_idx   = int(end_val)

                for i in range(start_idx, end_idx + 1, self.batch_size):
                    j = i + self.batch_size - 1
                    if j <= end_idx:
                        sequences.append((i, j))

            else:

                start_img = str(start_val).strip()
                end_img   = str(end_val).strip()

                start_prefix, start_frame_str = start_img.rsplit("_", 1)
                end_prefix,   end_frame_str   = end_img.rsplit("_", 1)
                assert start_prefix == end_prefix,\
                    f"起止文件不属于同一序列：{start_img} / {end_img}"

                start_idx = int(start_frame_str)
                end_idx   = int(end_frame_str)
                width     = len(start_frame_str)

                for i in range(start_idx, end_idx + 1, self.batch_size):
                    j = i + self.batch_size - 1
                    if j <= end_idx:
                        seq_start = f"{start_prefix}_{i:0{width}d}"
                        seq_end   = f"{start_prefix}_{j:0{width}d}"
                        sequences.append((seq_start, seq_end))

        return sequences


    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        vi_images, ir_images, flows = [], [], []


        if isinstance(start, int) and isinstance(end, int):

            prefix = ""
            frame_start, frame_end = start, end
            width = 0
        else:

            start, end = str(start), str(end)
            prefix, start_frame_str = start.rsplit("_", 1)
            prefix2, end_frame_str   = end.rsplit("_", 1)
            assert prefix == prefix2, f"起止文件前缀不一致: {start} / {end}"

            frame_start, frame_end = int(start_frame_str), int(end_frame_str)
            width = len(start_frame_str)


        for i in range(frame_start, frame_end + 1):
            if prefix:
                stem = f"{prefix}_{i:0{width}d}"
            else:
                stem = str(i)

            vi_image_path  = os.path.join(self.vi_path,  f"{stem}.png")
            ir_image_path  = os.path.join(self.ir_path,  f"{stem}.png")
            flow_path      = os.path.join(self.flow_path,f"{stem}.npy")

            vi_image = cv2.imread(vi_image_path, cv2.IMREAD_GRAYSCALE)
            ir_image = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
            flow     = np.load(flow_path).transpose(2, 0, 1)



            vi_image = np.expand_dims(vi_image.astype(np.float32) / 255.0, axis=0)
            ir_image = np.expand_dims(ir_image.astype(np.float32) / 255.0, axis=0)

            vi_images.append(vi_image)
            ir_images.append(ir_image)
            flows.append(flow)

        vi_images = np.stack(vi_images, axis=0)
        ir_images = np.stack(ir_images, axis=0)
        flows     = np.stack(flows, axis=0)

        return vi_images, ir_images, flows



class FusionDataset(Dataset):
    def __init__(self, vi_path, ir_path, transform=None):
        self.transform = transform

        self.vi_images = sorted(
            glob.glob(os.path.join(vi_path, '*.jpg')) +
            glob.glob(os.path.join(vi_path, '*.png'))
        )
        self.ir_images = sorted(
            glob.glob(os.path.join(ir_path, '*.jpg')) +
            glob.glob(os.path.join(ir_path, '*.png'))
        )

    def __len__(self):
        return len(self.vi_images)

    def __getitem__(self, idx):

        vi_image = cv2.imread(self.vi_images[idx], cv2.IMREAD_GRAYSCALE)
        ir_image = cv2.imread(self.ir_images[idx], cv2.IMREAD_GRAYSCALE)

        vi_image = cv2.resize(vi_image, (256,256))
        ir_image = cv2.resize(ir_image, (256,256))

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)

        return vi_image, ir_image


class RgbTestDataset(Dataset):
    def __init__(self, vi_path, ir_path, transform=None):
        self.transform = transform

        self.vi_images = sorted(glob.glob(os.path.join(vi_path, '*.jpg')) + glob.glob(os.path.join(vi_path, '*.png')) + glob.glob(os.path.join(vi_path, '*.bmp')))
        self.ir_images = sorted(glob.glob(os.path.join(ir_path, '*.jpg')) + glob.glob(os.path.join(ir_path, '*.png')) + glob.glob(os.path.join(ir_path, '*.bmp')))

    def __len__(self):
        return len(self.vi_images)

    def __getitem__(self, idx):

        vi_image = cv2.imread(self.vi_images[idx], cv2.IMREAD_COLOR)
        ir_image = cv2.imread(self.ir_images[idx], cv2.IMREAD_GRAYSCALE)

        vi_image_name = os.path.basename(self.vi_images[idx])
        ir_image_name = os.path.basename(self.ir_images[idx])

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)

        return vi_image, ir_image, vi_image_name, ir_image_name


class YTestDataset(Dataset):
    def __init__(self, vi_path, ir_path, transform=None):
        self.transform = transform

        self.vi_images = sorted(glob.glob(os.path.join(vi_path, '*.jpg')) + glob.glob(os.path.join(vi_path, '*.png')) + glob.glob(os.path.join(vi_path, '*.bmp')))
        self.ir_images = sorted(glob.glob(os.path.join(ir_path, '*.jpg')) + glob.glob(os.path.join(ir_path, '*.png')) + glob.glob(os.path.join(ir_path, '*.bmp')))

    def __len__(self):
        return len(self.ir_images)

    def __getitem__(self, idx):
        vi_image = cv2.imread(self.vi_images[idx], cv2.IMREAD_GRAYSCALE)
        ir_image = cv2.imread(self.ir_images[idx], cv2.IMREAD_GRAYSCALE)

        vi_image = vi_image.astype('float32')
        ir_image = ir_image.astype('float32')

        if self.transform:
            vi_image = self.transform(vi_image)
            ir_image = self.transform(ir_image)


        return vi_image, ir_image



import args
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def show_images(vi_image, ir_image, save_path, idx):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(vi_image, cmap='gray')
    ax1.set_title('VI Image')
    ax2.imshow(ir_image, cmap='gray')
    ax2.set_title('IR Image')
    plt.savefig(os.path.join(save_path, f"sample_{idx}.png"))
    plt.close(fig)

if __name__ == "__main__":

    vi_path = args.train_vi_path
    ir_path = args.train_ir_path
    save_path = "."

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    fusion_dataset = FusionDataset(vi_path, ir_path, transform=transform)
    fusion_dataloader = DataLoader(fusion_dataset, batch_size=5, shuffle=True)

    for i, (vi_image, ir_image) in enumerate(fusion_dataloader):

        print(ir_image.shape)
