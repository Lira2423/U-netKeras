import os
import h5py
from Utils import print_progress_bar, normalize_array


def minimize(files, output_dir, band_keys, indecies=None, band_names=None, skip=True):
    """
    Creates new hdf5 files with max compression and specified datasets only
    :param files: list of full paths to files
    :param output_dir: directory for minimized files
    :param band_keys: original file hdf keys to be kept
    :param indecies: optional, indecies to be saved
    :param band_names: optional, used to save datasets under new name
    :param skip: if set to True checks output directory for minimized files
    :return: None
    """
    if indecies is None:
        indecies = []
    if band_names is None:
        band_names = []

    print_progress_bar(0, len(files), prefix="Minimizing", suffix="Complete")
    for i, file in enumerate(files):
        filename = file.split(os.path.sep)[-1]
        if skip:
            if os.path.isfile(os.path.join(output_dir, filename)):
                print_progress_bar(i + 1, len(files), prefix="Minimizing", suffix="Complete")
                continue

        with h5py.File(file, "r+") as hdf5:
            with h5py.File(os.path.join(output_dir, filename), "w") as output:
                for num, key in enumerate(band_keys):
                    try:
                        band = normalize_array(hdf5[key][:, :])
                    except KeyError:
                        print(f"No key found:{key} skipping")
                        continue
                    try:
                        name = band_names[num]
                    except IndexError:
                        name = key.split("/")[-1]
                        print(f"No name for band {key} found, saving in root under name:{name}")
                    output.create_dataset(name, data=band,
                                          chunks=True, compression="gzip", compression_opts=9)

                for key in indecies:
                    try:
                        data = hdf5[key][:, :]
                    except KeyError:
                        print(f"No key found:{key} skipping")
                        continue
                    output.create_dataset(key, data=data,
                                          chunks=True, compression="gzip", compression_opts=9)
        print_progress_bar(i + 1, len(files), prefix="Minimizing", suffix="Complete")
