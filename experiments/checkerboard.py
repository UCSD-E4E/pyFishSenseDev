# %%
from pathlib import Path

import cv2
from pyaqua3ddev.image.image_processors import RawProcessor
from skimage.util import img_as_ubyte
from tqdm import tqdm

# %%
# rows = 14
# cols = 10

rows = 17
cols = 24

# %%
input_directory = Path(
    "/home/chris/data/2025.02.27.FishSense.Canyonview/ED-00/FSL-07D/Calibio"
)
files = list(input_directory.rglob("*.ORF"))

output_directory = Path(
    "/home/chris/data/2025.02.27.FishSense.Canyonview/output/FSL-07D/Calibio"
)
output_directory.mkdir(exist_ok=True, parents=True)


# %%
raw_processor = RawProcessor()

# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for idx, file in enumerate(tqdm(files)):
    image = img_as_ubyte(raw_processor.process(file))

    target = (
        output_directory
        / file.parent.relative_to(input_directory)
        / "rough"
        / f"{file.stem}.png"
    )
    target.parent.mkdir(exist_ok=True, parents=True)

    # convert to grayscale
    gray = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY)

    # find the checkerboard
    ret, corners_rough = cv2.findChessboardCorners(gray, (rows, cols), None)

    debug_image = cv2.drawChessboardCorners(
        image.copy(), (rows, cols), corners_rough.get(), ret
    )
    cv2.imwrite(target.absolute().as_posix(), debug_image)

    if not ret:
        continue

    target = (
        output_directory
        / file.parent.relative_to(input_directory)
        / "refined"
        / f"{file.stem}.png"
    )
    target.parent.mkdir(exist_ok=True, parents=True)

    # Convolution size used to improve corner detection. Don't make this too large.
    conv_size = (11, 11)

    # opencv can attempt to improve the checkerboard coordinates
    corners = cv2.cornerSubPix(gray, corners_rough, conv_size, (-1, -1), criteria)

    debug_image = cv2.drawChessboardCorners(
        image.copy(), (rows, cols), corners.get(), ret
    )
    cv2.imwrite(target.absolute().as_posix(), debug_image)
