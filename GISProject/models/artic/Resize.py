import h5py
from PIL import Image
import numpy as np


def resize(files, output_directory, resampler=Image.LANCZOS):
    """
    Resizes Arctic visible bands, saves only IR and VIS images with GEO data
    :param files: list of paths to files
    :param output_directory: output directory for new files
    :param resample: PIL resampler to be used
    :return: None
    """
    layers = ['B1_0.57', 'B2_0.72', 'B3_0.86', 'BT4_3.75', 'BT5_6.35',
              'BT6_8.00', 'BT7_8.70', 'BT8_9.7', 'BT9_10.7', 'BT10_11.7', "BT_Lon", "BT_Lat"]
    for f in files:

        size = (2784, 2784)
        file_name = f.split('\\')[-1]

        with h5py.File(f, 'r+') as original:
            with h5py.File(output_directory + '\\' + file_name, 'w') as output:
                for layer in layers:
                    shape = original[layer][:].shape
                    if shape != size:
                        resized = Image.fromarray(original[layer][:]).resize(size, resample=resampler)
                        resized = np.array(resized)
                        resized[resized < 1] = 0
                        resized = resized / np.amax(resized)
                        output.create_dataset(layer, data=resized, chunks=True,
                                              compression="gzip", compression_opts=9)
                    else:
                        no_resize = original[layer][:]
                        no_resize = no_resize / np.amax(no_resize)
                        output.create_dataset(layer, data=no_resize, chunks=True,
                                              compression="gzip", compression_opts=9)

