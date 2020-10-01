import numpy as np

from cv2 import imread, imwrite
from pathlib import Path

for p in Path("/home/crvenpaka/ftn/Oftalmologija/segmentacija-mreze/IterNet/data/DROPS/training/1st_manual/tmp").glob("**/*"):
    img = imread(str(p))
    h, w, _ = img.shape
    out = np.ndarray((h, w), dtype=np.uint8)
    for x in range(0, h):
        for y in range(0, w):
            if img[x,y,0] > 127:
                out[x,y] = 255
            else:
                out[x,y] = 0

    imwrite(str(p.parent.parent / p.name), out)