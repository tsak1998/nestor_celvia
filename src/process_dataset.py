from pathlib import Path
from openslide import OpenSlide
import h5py
from tqdm import tqdm

from pre_processor import extract_tiles

raw_slide_pth = Path('../ovarian_tissue_data/')
tile_metadata_pth = Path('tile_metadata')
tiles_pth = Path('tiles')
tile_size = 256


def save_patches_to_hdf5(tile_metadata, slide_id):
    with h5py.File(tile_metadata_pth / f"{slide_id}.hdf5", "w") as f:
        for i, patch in enumerate(tile_metadata):
            grp = f.create_group(f"patch_{i}")
            grp.attrs["x"] = patch["x"]
            grp.attrs["y"] = patch["y"]
            grp.attrs["contains_tissue"] = patch["contains_tissue"]


def process_slide(slide_pth: Path) -> None:

    slide_id = slide_pth.stem

    svs_image = OpenSlide(next(slide_pth.glob('*.svs')))
    tile_metadata = extract_tiles(svs_image=svs_image,
                                  tile_size=tile_size,
                                  slide_id=slide_id)

    # save_patches_to_hdf5(tile_metadata, slide_id)


if __name__ == '__main__':
    for slide_pth in tqdm(raw_slide_pth.glob('*')):
        if slide_pth.is_dir():

            process_slide(slide_pth=slide_pth)
