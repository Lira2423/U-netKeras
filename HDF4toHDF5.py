from pyhdf.SD import SD, SDC
import os
import h5py
import time

dir_input = "D:/College/Data/Terra_MODIS/2020.01.01/MOD021KM/"
dir_output = "D:/College/Data/Terra_MODIS/2020.01.01/MOD021KM_HDF5/"

Image_data = ["EV_250_Aggr1km_RefSB", "EV_500_Aggr1km_RefSB", "EV_1KM_RefSB", "EV_1KM_Emissive"]
GEO = ["Latitude", "Longitude"]

if __name__ == '__main__':
    start_all = time.time()
    f_not_filtered = []
    for (dirpath, dirnames, filenames) in os.walk(dir_input):
        f_not_filtered.extend(filenames)
        break
    f_filtered = list(filter(lambda f: f.endswith(".hdf"), f_not_filtered))

    for file in f_filtered:
        start_file = time.time()
        print(file)
        file_path = dir_input + file
        hdf4 = SD(file_path, SDC.READ)
        with h5py.File(dir_output+file+".h5", "w") as hdf5:
            bandGroup = hdf5.create_group("Bands")
            geoGroup = hdf5.create_group("GEO")

            for key in Image_data:
                data3D = hdf4.select(key)
                band_names = data3D.band_names.split(",")
                BandI = 0
                for layer in range(data3D.dim(0).length()):
                    data = data3D[layer, :, :]
                    bandGroup.create_dataset(f"{key}_Band_{band_names[BandI]}", data=data, chunks=True, compression="gzip", compression_opts=7)
                    BandI += 1
            for key in GEO:
                data = hdf4.select(key)[:, :]
                geoGroup.create_dataset(f"{key}", data=data, chunks=True, compression="gzip", compression_opts=7)
        end_file = time.time()
        print(f"Time:{end_file - start_file}")

    end_all = time.time()
    print(f"\n\n Runtime:{end_all-start_all}")
