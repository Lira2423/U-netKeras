import numpy as np
import pyhdf.SD
from pyhdf.SD import SD, SDC
import os
import h5py
from Utils import print_progress_bar


def convert_hdf(files, output_dir):
    """
    Converts MODIS files to HDF5 format + adds a day/night flag in filename
    :param files: list of full paths to files
    :param output_dir: directory for converted files
    :return: None
    """
    Image_data = ["EV_250_Aggr1km_RefSB", "EV_500_Aggr1km_RefSB", "EV_1KM_RefSB", "EV_1KM_Emissive"]
    GEO = ["Latitude", "Longitude", "SensorZenith", "SensorAzimuth", "SolarZenith", "SolarAzimuth", "Height", "Range"]
    Band_attributes = ["radiance_scales", "radiance_offsets", "reflectance_scales", "reflectance_offsets"]

    print_progress_bar(0, len(files), prefix="Converting")
    for i, file_path in enumerate(files):
        file_name = file_path.split(os.path.sep)[-1]
        hdf4 = SD(file_path, SDC.READ)

        total_scans = hdf4.attributes().get("Number of Scans")
        day_scans = hdf4.attributes().get("Number of Day mode scans")
        night_scans = hdf4.attributes().get("Number of Night mode scans")
        incomplete_scans = hdf4.attributes().get("Incomplete Scans")

        flag = ".M"
        if total_scans == day_scans:
            flag = ".D"
        if total_scans == night_scans:
            flag = ".N"

        file_to_create = f"{output_dir}{os.path.sep}{file_name[:-4]}{flag}.h5"
        if os.path.exists(file_to_create):
            print_progress_bar(i + 1, len(files), prefix="Converting", suffix=f"{i + 1} out of {len(files)}")
            continue

        with h5py.File(file_to_create, "w") as hdf5:
            hdf5.attrs.create("Number of Scans", total_scans)
            hdf5.attrs.create("Number of Day mode scans", day_scans)
            hdf5.attrs.create("Number of Night mode scans", night_scans)
            hdf5.attrs.create("Incomplete Scans", incomplete_scans)

            bandGroup = hdf5.create_group("Bands")
            geoGroup = hdf5.create_group("GEO")

            for key in Image_data:
                data3D = hdf4.select(key)
                band_names = data3D.band_names.split(",")

                for attr in Band_attributes:
                    try:
                        attribute_data = data3D.attributes().get(attr)
                        attribute_data = np.array(attribute_data)
                        if attribute_data.dtype != "O":
                            bandGroup.attrs.create(f"{key}_{attr}", attribute_data, dtype=np.float32)
                    except pyhdf.SD.HDF4Error:
                        continue

                BandI = 0
                for layer in range(data3D.dim(0).length()):
                    data = data3D[layer, :, :]
                    data = data / np.amax(data)
                    bandGroup.create_dataset(f"{key}_Band_{band_names[BandI]}", data=data, chunks=True,
                                                       compression="gzip", compression_opts=9)
                    BandI += 1

            for key in GEO:
                data = hdf4.select(key)[:, :]
                geoGroup.create_dataset(f"{key}", data=data, chunks=True, compression="gzip", compression_opts=9)
        print_progress_bar(i + 1, len(files), prefix="Converting", suffix=f"{i + 1} out of {len(files)}")

