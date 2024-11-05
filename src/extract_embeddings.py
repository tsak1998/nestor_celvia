import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

embeddings_type = "pretrained"

base_tile_pth = Path("/Users/tsakalis/ntua/nestor/nestor_celvia/src/tiles/")
base_embeddings_pth = Path(
    f"/Users/tsakalis/ntua/nestor/nestor_celvia/src/extracted_embeddings_{embeddings_type}/"
)
base_embeddings_pth.mkdir(parents=True, exist_ok=True)


class SampleGroupedDataset(Dataset):

    def __init__(self, base_tile_path, sample_ids, transform=None):
        self.samples = []
        self.transform = transform

        # Collect images grouped by sample directory
        for sample_id in sample_ids:
            sample_path = base_tile_path / sample_id
            image_paths = list(sample_path.glob('*.jpeg'))[:2]
            if image_paths:
                self.samples.append((sample_id.stem, image_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # breakpoint()
        sample_id, image_paths = self.samples[idx]

        images = []
        for img_path in image_paths:
            img_3c = np.array(Image.open(img_path))
            if self.transform:
                img_3c = self.transform(img_3c)
            img_tensor = torch.tensor(img_3c, dtype=torch.float32).permute(
                2, 0, 1)  # (C, H, W)
            images.append(img_tensor)
        image_tensor = torch.stack(images)
        return sample_id, image_tensor


# Parameters
# base_tile_path = Path('/path/to/base/tile')

device = "cuda:1" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    MedSAM_CKPT_PATH = "/Users/tsakalis/ntua/nestor/nestor_celvia/medsam_pretrained_weights/medsam_vit_b.pth"

    medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    batch_size = 1  # Hacky way to get all slide tiles into same batch
    sample_ids = list(base_tile_pth.glob('*'))

    # Create dataset and dataloader
    dataset = SampleGroupedDataset(base_tile_pth, sample_ids, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, total=len(dataloader)):
        sample_id, batch_tensor = batch

        with torch.no_grad():

            batch_tensor = batch_tensor.to(device).squeeze(0)

            image_embedding = medsam_model.image_encoder(batch_tensor)

            torch.save(image_embedding.to('cpu'),
                       base_embeddings_pth / f"{sample_id[0]}.pt")
        break
