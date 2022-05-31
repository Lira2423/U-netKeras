import h5py
import numpy as np
from Utils import print_progress_bar, normalize_array


# MODIS green:band 4,12 nir:band 2,16
def calculate_NDWI(files, green_bands, nir_bands, cutoff=0.5, maskOnly=True):
    """
    Calculates NDWI based off provided bands

    :param files: full path filelist
    :param green_bands: list of green band keys, if more than 1 calculates mean between all
    :param nir_bands: list of nir band keys
    :param maskOnly: if set to true, uses cutoff value to create final mask, otherwise saves index as is
    :param cutoff: cutoff value
    :return: None
    """
    calculate_index("NDWI", files, green_bands, nir_bands, cutoff=cutoff, maskOnly=maskOnly)


# MODIS swir:band 6
def calculate_NDSI(files, green_bands, swir_bands, cutoff=0.1, maskOnly=True):
    """
    Calculates NDWI based off provided bands

    :param files: full path filelist
    :param green_bands: list of green band keys, if more than 1 calculates mean between all
    :param swir_bands: list of swir band keys
    :param maskOnly: if set to true, uses cutoff value to create final mask, otherwise saves index as is
    :param cutoff: cutoff value
    :return: None
    """
    calculate_index("NDSI", files, green_bands, swir_bands, cutoff=cutoff, maskOnly=maskOnly)


def calculate_index(indexName, files, band_list1, band_list2, cutoff=0.0, maskOnly=True):
    """
    Calculates specified index based off provided bands as:
    (band1 - band2) / (band1 + band2).
    Creates dataset with specified name.

    :param indexName: name of dataset to create
    :param files: full path list
    :param band_list1: list of band keys, if more than 1 calculates mean between all
    :param band_list2: list of band keys
    :param maskOnly: if set to true, uses cutoff value to create final mask, otherwise saves index as is
    :param cutoff: cutoff value
    :return: None
    """

    if len(band_list1) == 0 or len(band_list2) == 0:
        print("No bands specified")
        return

    print(f"Total items:{len(files)}")
    print_progress_bar(0, len(files), prefix=indexName, suffix="Complete")

    for i, file in enumerate(files):
        with h5py.File(file, "r+") as hdf5:
            shape = hdf5[band_list1[0]][:, :].shape
            mean_first = np.zeros(shape)
            for band in band_list1:
                mean_first = mean_first + hdf5[band][:, :]
            mean_first = mean_first / len(band_list1)

            mean_second = np.zeros(shape)
            for band in band_list2:
                mean_second = mean_second + hdf5[band][:, :]
            mean_second = mean_second / len(band_list2)

            first = np.subtract(mean_first, mean_second)
            second = np.add(mean_first, mean_second)

            INDEX = np.divide(first, second, where=second != 0)

            if maskOnly:
                INDEX[INDEX < cutoff] = 0
                INDEX[INDEX >= cutoff] = 1
                INDEX = INDEX.astype(int)

            try:
                del hdf5[indexName]
            except KeyError:
                pass
            hdf5.create_dataset(indexName, data=INDEX, chunks=True, compression="gzip", compression_opts=9)

        print_progress_bar(i + 1, len(files), prefix=indexName, suffix="Complete")

# MODIS blue:band 3
def calculate_EVI(files, red_bands, NIR_bands, blue_bands, cutoff=0.0, maskOnly=False):
    """
    Creates dataset with calculated EV index.

    :param files: full path list
    :param red_bands: list of red band keys, if more than 1 calculates mean between all
    :param NIR_bands: list of NIR band keys
    :param blue_bands: list of blue band keys
    :param maskOnly: if set to true, uses cutoff value to create final mask, otherwise saves index as is
    :param cutoff: cutoff value
    :return: None
    """

    if len(red_bands) == 0 or len(NIR_bands) == 0 or len(blue_bands) == 0:
        print("No bands specified")
        return

    print(f"Total items:{len(files)}")
    print_progress_bar(0, len(files), prefix="EVI", suffix="Complete")

    for i, file in enumerate(files):
        with h5py.File(file, "r+") as hdf5:
            shape = hdf5[red_bands[0]][:, :].shape

            mean_first = np.zeros(shape)
            for band in red_bands:
                mean_first = mean_first + hdf5[band][:, :]
            mean_first = mean_first / len(red_bands)

            mean_second = np.zeros(shape)
            for band in NIR_bands:
                mean_second = mean_second + hdf5[band][:, :]
            mean_second = mean_second / len(NIR_bands)

            mean_third = np.zeros(shape)
            for band in blue_bands:
                mean_third = mean_third + hdf5[band][:, :]
            mean_third = mean_third / len(blue_bands)

            first = mean_second - mean_first
            second = mean_second + (mean_first * 6) - (7.5 * mean_third) + 1
            INDEX = 2.5 * np.divide(first, second, where=second != 0)

            if maskOnly:
                INDEX[INDEX < cutoff] = 0
                INDEX[INDEX >= cutoff] = 1
                INDEX = INDEX.astype(int)

            try:
                del hdf5["EVI"]
            except KeyError:
                pass
            hdf5.create_dataset("EVI", data=INDEX, chunks=True, compression="gzip", compression_opts=9)

        print_progress_bar(i + 1, len(files), prefix="EVI", suffix="Complete")

##MODIS ONLY
def cloud_mask(files):
    """
    Crates cloud masks by reflective brightness and temperature difference
    :param files: paths to converted MODIS files
    :return: None
    """
    print(f"Total items:{len(files)}")
    print_progress_bar(0, len(files), prefix="Cloud Mask", suffix="Complete")
    for i, f in enumerate(files):
        with h5py.File(f, "r+") as hdf:
            CH1 = normalize_array(hdf["Bands/EV_250_Aggr1km_RefSB_Band_1"][:, :])
            CH3 = normalize_array(hdf["Bands/EV_500_Aggr1km_RefSB_Band_3"][:, :])

            CH29 = normalize_array(hdf["Bands/EV_1KM_Emissive_Band_29"][:, :])
            CH31 = normalize_array(hdf["Bands/EV_1KM_Emissive_Band_31"][:, :])

            conditionR = CH1 >= 0.2
            conditionB = CH3 >= 0.2

            conditionT = np.logical_and(CH29 <= 0.5, CH31 <= 0.4)

            mask = conditionR * conditionB * conditionT
            try:
                del hdf["clouds"]
            except KeyError:
                pass
            hdf.create_dataset("clouds", data=mask * 255, chunks=True, compression="gzip", compression_opts=9)

        print_progress_bar(i + 1, len(files), prefix="Cloud Mask", suffix="Complete")

##MODIS ONLY
def cloud_mask2(files):
    """
    Crates cloud masks by adding and subtracting compound images, works better in snowy areas
    :param files: paths to converted MODIS files
    :return: None
    """
    print(f"Total items:{len(files)}")
    print_progress_bar(0, len(files), prefix="Cloud Mask", suffix="Complete")
    for i, f in enumerate(files):
        with h5py.File(f, "r+") as hdf:
            first = np.array([normalize_array(hdf["Bands/EV_500_Aggr1km_RefSB_Band_5"][:, :]),
                              normalize_array(hdf["Bands/EV_250_Aggr1km_RefSB_Band_1"][:, :]),
                              normalize_array(hdf["Bands/EV_500_Aggr1km_RefSB_Band_3"][:, :])])

            second = np.array([normalize_array(hdf["Bands/EV_1KM_RefSB_Band_26"][:, :]),
                               normalize_array(hdf["Bands/EV_250_Aggr1km_RefSB_Band_2"][:, :]),
                               normalize_array(hdf["Bands/EV_1KM_Emissive_Band_31"][:, :])])

            compound = first + second
            compound[compound > 1.0] = 1.0

            third = np.array([normalize_array(hdf["Bands/EV_1KM_Emissive_Band_31"][:, :]),
                              normalize_array(hdf["Bands/EV_500_Aggr1km_RefSB_Band_4"][:, :]),
                              normalize_array(hdf["Bands/EV_1KM_RefSB_Band_26"][:, :])])

            fourth = np.array([normalize_array(hdf["Bands/EV_250_Aggr1km_RefSB_Band_2"][:, :]),
                               normalize_array(hdf["Bands/EV_500_Aggr1km_RefSB_Band_4"][:, :]),
                               normalize_array(hdf["Bands/EV_1KM_RefSB_Band_26"][:, :])])

            final = compound - third - fourth
            final[final < 0.0] = 0.0
            final = final[0]
            try:
                del hdf["cloud2"]
            except KeyError:
                pass
            hdf.create_dataset("cloud2", data=final, chunks=True, compression="gzip", compression_opts=9)
        print_progress_bar(i + 1, len(files), prefix="Cloud Mask", suffix="Complete",)

##MODIS ONLY
def snow_mask(files):
    """
    Crates snow masks with help of NDSI, NDWI and pre-made cloud masks
    :param files: paths to converted MODIS files
    :return: None
    """

    print(f"Total items:{len(files)}")
    print_progress_bar(0, len(files), prefix="Snow Mask", suffix="Complete")

    for i, f in enumerate(files):
        with h5py.File(f, "r+") as hdf:
            NDSI = hdf["NDSI"][:, :]
            NDWI = hdf["NDWI"][:, :]
            CLOUD = hdf["cloud2"][:, :]

            NDSI[NDWI >= 0.5] = 0.0
            NDSI[CLOUD >= 0.01] = 0.0
            NDSI[NDSI >= 0.1] = 1.0
            NDSI[NDSI < 0.1] = 0.0

            try:
                del hdf["snow"]
            except KeyError:
                pass
            hdf.create_dataset("snow", data=NDSI.astype(int), chunks=True, compression="gzip", compression_opts=9)
        print_progress_bar(i + 1, len(files), prefix="Snow Mask", suffix="Complete")
