from Utils import filter_files_paths, create_preview, create_preview_W
from models.Split import split_in_chunks
from models.Mask import *
from models.modis.Minimise import *
from models.modis.HDF4toHDF5 import *

modis_minKeys = ["Bands/EV_1KM_Emissive_Band_20", "Bands/EV_1KM_Emissive_Band_27",
                 "Bands/EV_1KM_Emissive_Band_28", "Bands/EV_1KM_Emissive_Band_29",
                 "Bands/EV_1KM_Emissive_Band_27", "Bands/EV_1KM_Emissive_Band_31",
                 "Bands/EV_250_Aggr1km_RefSB_Band_1", "Bands/EV_500_Aggr1km_RefSB_Band_4",
                 "Bands/EV_500_Aggr1km_RefSB_Band_3"]
modis_minNames = ["Band_20", "Band_27", "Band_28", "Band_29", "Band_30", "Band_31",
                     "Red", "Green", "Blue"]
modis_misc = ["NDWI", "NDSI", "NDVI", "EVI", "clouds", "cloud2", "snow"]

modis_green = ["Bands/EV_500_Aggr1km_RefSB_Band_4", "Bands/EV_1KM_RefSB_Band_12"]
modis_red = ["Bands/EV_250_Aggr1km_RefSB_Band_1"]
modis_blue = ["Bands/EV_500_Aggr1km_RefSB_Band_3"]

modis_SWIR = ["Bands/EV_500_Aggr1km_RefSB_Band_6"]
modis_NIR = ["Bands/EV_250_Aggr1km_RefSB_Band_2"]

arctic_bands = ['B1_0.57', 'B2_0.72', 'B3_0.86', 'BT4_3.75', 'BT5_6.35',
                'BT6_8.00', 'BT7_8.70', 'BT8_9.7', 'BT9_10.7']

def diplom():
    # Setup folders
    original_modis_files_dir = os.path.normpath("D:\\college\\Data\\Diplom2\\MOD_ORIG")
    hdf5_output = os.path.normpath("D:\\College\\Data\\Diplom2\\MOD_HDF5")
    min_output = os.path.normpath("D:\\College\\Data\\Diplom2\\DATASET")
    chunks_output = os.path.normpath("D:\\College\\Data\\Diplom2\\DATASET\\Chunks")

    # Convert to HDF5
    files = filter_files_paths(original_modis_files_dir, ".hdf")
    convert_hdf(files, hdf5_output)

    # Calculate indecies, create masks
    files = filter_files_paths(hdf5_output, ".h5")
    calculate_NDWI(files, modis_green, modis_NIR, maskOnly=False, cutoff=0.62)
    calculate_NDSI(files, modis_green, modis_SWIR, maskOnly=False)
    calculate_EVI(files, modis_red, modis_NIR, modis_blue, maskOnly=False)
    cloud_mask(files)
    cloud_mask2(files)
    snow_mask(files)

    # Save data for learning only
    minimize(files, min_output, modis_minKeys,
             indecies=modis_misc, band_names=modis_minNames, skip=False)

    # Create previews
    files = filter_files_paths(min_output, ".h5")
    for i, f in enumerate(files):
        imageC = create_preview_W(f, "cloud2")
        imageS = create_preview(f, "snow")
        image = create_preview(f, "Red", "Green", "Blue")

        image.save(f[:-3] + ".png", format="png", optimize=True)
        imageC.save(f[:-3] + "_C.png", format="png", optimize=True)
        imageS.save(f[:-3] + "_S.png", format="png", optimize=True)

    # Create dataset
    split_in_chunks(files, chunks_output, modis_minNames + modis_misc, skip=False)


if __name__ == '__main__':
    diplom()
