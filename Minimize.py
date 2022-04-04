import os
import h5py
import numpy as np

dir_input = "D:/College/Data/Terra_MODIS/2020.01.01/MOD021KM_HDF5/"
dir_mid = "D:/College/Data/Terra_MODIS/2020.01.01/Dataset/"
dir_final = "D:/College/Data/Terra_MODIS/2020.01.01/Dataset/Chunks/"

pixels = 1024


def split_dataset():
    f_not_filtered = []
    for (dirpath, dirnames, filenames) in os.walk(dir_mid):
        f_not_filtered.extend(filenames)
        break
    f_filtered = list(filter(lambda f: f.endswith(".h5"), f_not_filtered))

    chunks = 2

    for file in f_filtered:
        filename = file[:-7]
        print(filename)
        with h5py.File(dir_mid + file, "r+") as original:
            directory = f"{dir_final}/{filename.replace('.', '_')}"
            os.makedirs(directory, exist_ok=True)

            n = 0
            for i in range(0, chunks):
                for j in range(0, chunks):
                    with h5py.File(f"{directory}/{str(n).zfill(3)}.h5", "w") as output_file:
                        for band in original.keys():
                            chunk = original[band][pixels * i:pixels + pixels * i, pixels * j:pixels + pixels * j]
                            if chunk.shape != (pixels, pixels):
                                expanded = np.zeros((pixels, pixels))
                                expanded[:chunk.shape[0], :chunk.shape[1]] = chunk
                                chunk = expanded
                            output_file.create_dataset(band, data=chunk,
                                                       chunks=True, compression="gzip", compression_opts=7)

                    n += 1


def minimize():
    f_not_filtered = []
    for (dirpath, dirnames, filenames) in os.walk(dir_input):
        f_not_filtered.extend(filenames)

    f_filtered = list(filter(lambda f: f.endswith(".h5"), f_not_filtered))

    for file in f_filtered:
        print(file)
        with h5py.File(dir_input + file, "r+") as hdf5:
            with h5py.File(dir_mid + file, "w") as output:
                output.create_dataset("Red", data=hdf5["Bands/EV_250_Aggr1km_RefSB_Band_1"][:, :],
                                      chunks=True, compression="gzip", compression_opts=7)
                output.create_dataset("Green", data=hdf5["Bands/EV_500_Aggr1km_RefSB_Band_4"][:, :],
                                      chunks=True, compression="gzip", compression_opts=7)
                output.create_dataset("Blue", data=hdf5["Bands/EV_500_Aggr1km_RefSB_Band_3"][:, :],
                                      chunks=True, compression="gzip", compression_opts=7)
                output.create_dataset("Band_20", data=hdf5["Bands/EV_1KM_Emissive_Band_20"][:, :],
                                      chunks=True, compression="gzip", compression_opts=7)

                for i in range(27, 31):
                    band = hdf5[f"Bands/EV_1KM_Emissive_Band_{i}"][:, :]
                    output.create_dataset(f"Band_{i}", data=band,
                                          chunks=True, compression="gzip", compression_opts=7)

                NDWI = hdf5["NDWI"][:, :]
                output.create_dataset("NDWI", data=NDWI, chunks=True, compression="gzip", compression_opts=7)


if __name__ == '__main__':
    minimize()
    split_dataset()
