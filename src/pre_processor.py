from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from skimage.morphology import closing, square
from skimage import img_as_ubyte
from openslide import OpenSlide
from PIL import Image

level = 0
tiles_path = Path('F:/process_data/tiles')
msk_pth = Path('F:/process_data/binary_masks')


class TileData(TypedDict):
    x: int
    y: int
    contains_tissue: bool


def create_tissue_mask(img: np.ndarray) -> np.ndarray:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hsv_img[:, :, 1]

    _, mask = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY)

    # mask = closing(mask, square(5))
    mask = img_as_ubyte(mask)

    return mask


def image_tile_to_jpeg(img: np.ndarray, slide_path: Path) -> None:
    ...


def extract_tiles(
    svs_image: OpenSlide,
    tile_size: int,
    slide_id: str,
    mask_threshold: int = 210,
    save_tissue_tiles: bool = True,
) -> list[TileData]:

    slide_width, slide_height = svs_image.level_dimensions[level]

    tile_metadata: list[TileData] = []
    i, j = 0, 0

    tiles_output_pth = tiles_path / slide_id
    msk_output_pth = msk_pth / slide_id

    if save_tissue_tiles:
        tiles_output_pth.mkdir(parents=True, exist_ok=True)
        msk_output_pth.mkdir(parents=True, exist_ok=True)

    region_size = tile_size * tile_size
    height_mod = slide_height % tile_size
    width_mod = slide_width % tile_size
    for y in range(0, slide_height - height_mod, tile_size):
        i += 1
        for x in range(0, slide_width - width_mod, tile_size):
            j += 1

            import time 
            t1 = time.time()
            region = svs_image.read_region((x, y), level,
                                           (tile_size, tile_size))
            
            # print(time.time()-t1)

            region_ar = np.array(region.convert("RGB"))

            mask = create_tissue_mask(region_ar)

            # semi empty noisy tiles
            zero_pixels = np.count_nonzero(region_ar == 0)
            bright_pixels = np.count_nonzero(region_ar > 210)
            bw_condition = region_size != (zero_pixels + bright_pixels)

            if (np.mean(region_ar) < mask_threshold) and bw_condition:

                tile_metadata.append({"x": x, "y": y, "contains_tissue": True})
                if save_tissue_tiles:
                    region.convert("RGB").save(tiles_output_pth /
                                               f"{i}_{j}.jpeg")
                    Image.fromarray(mask).save(msk_output_pth /
                                               f"{i}_{j}.jpeg")

            else:
                tile_metadata.append({
                    "x": x,
                    "y": y,
                    "contains_tissue": False
                })

    return tile_metadata
