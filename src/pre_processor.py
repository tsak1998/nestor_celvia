from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from skimage.morphology import closing, square
from skimage import img_as_ubyte
from openslide import OpenSlide

level = 0
tiles_path = Path('tiles')


class TileData(TypedDict):
    x: int
    y: int
    contains_tissue: bool


def create_tissue_mask(img: np.ndarray) -> np.ndarray:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)

    _, mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)

    mask = closing(mask, square(5))
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

    if save_tissue_tiles:
        tiles_output_pth.mkdir(parents=True, exist_ok=True)

    for y in range(0, slide_height - slide_height % tile_size, tile_size):
        i += 1
        for x in range(0, slide_width - slide_width % tile_size, tile_size):
            j += 1
            region = svs_image.read_region((x, y), level,
                                           (tile_size, tile_size))

            region_ar = np.array(region.convert("RGB"))

            mask = create_tissue_mask(region_ar)

            flat_arr = np.array(region_ar).flatten()

            # semi empty noisy tiles
            bw_condition = flat_arr.shape[0] != (((flat_arr == 0).sum() +
                                                  (flat_arr > 210).sum()))

            if (np.mean(region_ar) < mask_threshold) and bw_condition:

                tile_metadata.append({"x": x, "y": y, "contains_tissue": True})
                if save_tissue_tiles:
                    region.convert("RGB").save(tiles_output_pth /
                                               f"{i}_{j}.jpeg")

            else:
                tile_metadata.append({
                    "x": x,
                    "y": y,
                    "contains_tissue": False
                })

    return tile_metadata
