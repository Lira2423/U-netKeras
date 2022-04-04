import os
import h5py
import numpy as np

# Взять среднее между 4 и 12
# Посчитать NDWI на среднем и 2 (или 16)

dir_input = "D:/College/Data/Terra_MODIS/2020.01.01/MOD021KM_HDF5/"
Band4 = "EV_500_Aggr1km_RefSB_Band_4"
Band12 = "EV_1KM_RefSB_Band_12"

Band2 = "EV_250_Aggr1km_RefSB_Band_2"
Band16 = "EV_1KM_RefSB_Band_16"


def calculate_ndwi():
    f_not_filtered = []
    for (dirpath, dirnames, filenames) in os.walk(dir_input):
        f_not_filtered.extend(filenames)
        break
    f_filtered = list(filter(lambda f: f.endswith(".h5"), f_not_filtered))

    for file in f_filtered:
        print(file)
        with h5py.File(dir_input + file, "r+") as hdf5:
            data4 = hdf5["Bands/" + Band4][:, :]
            data12 = hdf5["Bands/" + Band12][:, :]
            mean = np.nanmean(np.array([data4, data12]), axis=0)

            data2 = hdf5["Bands/" + Band2][:, :]
            NDWI = np.divide(np.subtract(mean, data2), np.add(mean, data2))
            NDWI[NDWI < 0.620] = 0
            NDWI[NDWI >= 0.620] = 255

            try:
                del hdf5["NDWI"]
            except KeyError:
                pass
            hdf5.create_dataset("NDWI", data=NDWI.astype(int), chunks=True, compression="gzip", compression_opts=7)


if __name__ == '__main__':
    calculate_ndwi()
    print("DONE")
