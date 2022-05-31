import os
import h5py
import numpy
import numpy as np
from PIL import Image


def filter_files_names(directory, ends_with):
    """
    Filters files by trailing substring
    :param directory: directory to filter
    :param ends_with: filter files by this trailing substring
    :return: List with file names in specified folder
    """
    f_not_filtered = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        f_not_filtered.extend(filenames)
        break
    return list(filter(lambda f: f.endswith(ends_with), f_not_filtered))


def filter_files_paths(directory, ends_with):
    """
    Filters files by trailing substring
    :param directory: directory to filter
    :param ends_with: filter files by this trailing substring
    :return: List with full paths to files in specified folder
    """
    f_not_filtered = []
    for root, dirs, filenames in os.walk(directory):
        for file in filenames:
            f_not_filtered.append(os.path.join(root, file))
        break
    return list(filter(lambda f: f.endswith(ends_with), f_not_filtered))


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def create_preview(file, red_band_key, green_band_key, blue_band_key):
    """
    Creates preview of specified satellite bands as RGB image
    :param file: full file path
    :param red_band_key: hdf key for red band
    :param green_band_key: hdf key for green band
    :param blue_band_key:  hdf key for blue band
    :return: PIL.Image
    """
    with h5py.File(file, "r") as hdf:
        shape = hdf[red_band_key][:, :].shape
        green = hdf[green_band_key][:, :]
        red = hdf[red_band_key][:, :]
        blue = hdf[blue_band_key][:, :]

        green = normalize_array(green)
        red = normalize_array(red)
        blue = normalize_array(blue)

        image = np.asarray([(red * 255), (green * 255), (blue * 255)])\
            .swapaxes(0, 2) \
            .swapaxes(0, 1)
        image = Image.fromarray(image.astype(numpy.uint8), mode="RGB")
        return image

def create_preview_W(file, band_key):
    """
    Creates preview of specified satellite bands as RGB image
    :param file: full file path
    :param red_band_key: hdf key for red band
    :param green_band_key: hdf key for green band
    :param blue_band_key:  hdf key for blue band
    :return: PIL.Image
    """
    with h5py.File(file, "r") as hdf:
        shape = hdf[band_key][:, :].shape
        band = hdf[band_key][:, :]
        green = normalize_array(band)


        image = np.asarray([(band * 255), (band * 255), (band * 255)])\
            .swapaxes(0, 2) \
            .swapaxes(0, 1)
        image = Image.fromarray(image.astype(numpy.uint8), mode="RGB")
        return image


def normalize_array(array):
    """
    Normalizes numpy array between 0 and 1
    :param array: Input numpy array
    :return: Normalized numpy array
    """
    array = array.astype("float32")
    minA = np.min(array)
    array = (array - minA)/np.ptp(array)
    return array
