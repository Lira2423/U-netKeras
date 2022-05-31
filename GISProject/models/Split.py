import math
import numpy as np
import h5py
import os
from Utils import print_progress_bar


def split_in_chunks(files, output_directory, bands, size=1024, skip=True):
    """
    Splits satellite images into chunks of specified size
    :param files: full path list
    :param output_directory: output directory
    :param bands: list of bands to be divided into chunks
    :param size: size of chunks in pixels
    :param skip: if true, skips previously skipped files
    :return: None
    """

    print_progress_bar(0, len(files), prefix="Cropping", suffix="Complete")
    for f_index, f in enumerate(files):
        filename = f.split(os.path.sep)[-1][:-3]
        pixels = size

        with h5py.File(f, "r") as file:
            wrong_size = False
            shape = None
            for layer in bands:
                try:
                    if shape is not None:
                        if shape != file[layer][:, :].shape:
                            print(f"{filename} has different size datasets skipping")
                            wrong_size = True
                    shape = file[layer][:, :].shape
                except KeyError:
                    continue

            if wrong_size:
                print_progress_bar(f_index + 1, len(files), prefix="Cropping", suffix="Complete")
                continue

            gridX = math.ceil(shape[0] / pixels)
            gridY = math.ceil(shape[1] / pixels)

            directory_to_create = f"{output_directory}{os.path.sep}{filename.replace('.', '_')}"
            if skip and os.path.exists(directory_to_create):
                files_ex = len(os.listdir(directory_to_create))
                if files_ex == gridX * gridY:
                    print_progress_bar(f_index + 1, len(files), prefix="Cropping", suffix="Complete")
                    continue

            os.makedirs(directory_to_create, exist_ok=True)

            n = 0
            for i in range(0, gridX):
                for j in range(0, gridY):
                    with h5py.File(f"{directory_to_create}{os.path.sep}{str(n).zfill(3)}.h5", "w") as output:
                        for layer in bands:
                            try:
                                chunk = file[layer][pixels*i:pixels+pixels*i, pixels*j:pixels+pixels*j]
                            except KeyError:
                                continue
                            if chunk.shape != (pixels, pixels):
                                expanded = np.zeros((pixels, pixels))
                                expanded[:chunk.shape[0], :chunk.shape[1]] = chunk
                                chunk = expanded
                            output.create_dataset(layer, data=chunk, chunks=True,
                                                  compression="gzip", compression_opts=9)
                    n += 1
        print_progress_bar(f_index + 1, len(files), prefix="Cropping", suffix="Complete")
