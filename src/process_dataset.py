import os
os.add_dll_directory(r'C:\openslide-bin-4.0.0.6-windows-x64\bin')
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openslide import OpenSlide
import h5py
from tqdm import tqdm
from pre_processor import extract_tiles
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import sys
sys.stderr = open(os.devnull, 'w')

raw_slide_pth = Path('F:/ovarian_data/')
tile_metadata_pth = Path('F:/process_data/tile_metadata')
tiles_pth = Path('F:/process_data/tiles')
tile_size = 1024


def save_patches_to_hdf5(tile_metadata, slide_id):
    with h5py.File(tile_metadata_pth / f"{slide_id}.hdf5", "w") as f:
        for i, patch in enumerate(tile_metadata):
            grp = f.create_group(f"patch_{i}")
            grp.attrs["x"] = patch["x"]
            grp.attrs["y"] = patch["y"]
            grp.attrs["contains_tissue"] = patch["contains_tissue"]


def process_slide(slide_pth: Path) -> None:
    slide_id = slide_pth.stem
    try:

        svs_image = OpenSlide(next(slide_pth.glob('*.svs')))

        tile_metadata = extract_tiles(svs_image=svs_image,
                                      tile_size=tile_size,
                                      slide_id=slide_id)

        save_patches_to_hdf5(tile_metadata, slide_id)
    except Exception as e:
        print('=================')
        print(e)
        print(slide_id)
        return None


if __name__ == '__main__':

    all_slides = [
        slide_pth for slide_pth in raw_slide_pth.glob('*')
        if slide_pth.is_dir()
    ]

    with Pool() as pool:
        for _ in tqdm(pool.imap(process_slide, all_slides),
                      total=len(all_slides)):
            pass
