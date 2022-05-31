import h5py
from netCDF4 import Dataset
import os


def nc_pac(ln, lt, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, name):
    new_name = name + '_1.nc'
    root_grp = Dataset(new_name, 'w', format='NETCDF4')
    root_grp.description = 'Example simulation data'
    ndim = b4.shape
    root_grp.createDimension('dimx', ndim[0])
    root_grp.createDimension('dimy', ndim[1])
    root_grp.createVariable('Band 1_0.57', 'd', ('dimx', 'dimy'))[:] = b1
    root_grp.createVariable('Band 2_0.72', 'd', ('dimx', 'dimy'))[:] = b2
    root_grp.createVariable('Band 3_0.86', 'd', ('dimx', 'dimy'))[:] = b3
    root_grp.createVariable('Band 4_3.75', 'd', ('dimx', 'dimy'))[:] = b4
    root_grp.createVariable('Band 5_6.35', 'd', ('dimx', 'dimy'))[:] = b5
    root_grp.createVariable('Band 6_8.00', 'd', ('dimx', 'dimy'))[:] = b6
    root_grp.createVariable('Band 7_8.70', 'd', ('dimx', 'dimy'))[:] = b7
    root_grp.createVariable('Band 8_9.7', 'd', ('dimx', 'dimy'))[:] = b8
    root_grp.createVariable('Band 9_10.7', 'd', ('dimx', 'dimy'))[:] = b9
    root_grp.createVariable('Band 10_11.7', 'd', ('dimx', 'dimy'))[:] = b10
    root_grp.createVariable('Longitude', 'd', ('dimx', 'dimy'))[:] = ln
    root_grp.createVariable('Latitude', 'd', ('dimx', 'dimy'))[:] = lt


def convert(input_directory):
    files = (os.listdir(input_directory))
    for i in files:
        filename = input_directory + i
        print(filename)

        try:
            with h5py.File(filename, 'r') as f:
                bt1 = f['B1_0.57'][::4, ::4]
                bt2 = f['B2_0.72'][::4, ::4]
                bt3 = f['B3_0.86'][::4, ::4]
                bt4 = f['BT4_3.75'][:]
                bt5 = f['BT5_6.35'][:]
                bt6 = f['BT6_8.00'][:]
                bt7 = f['BT7_8.70'][:]
                bt8 = f['BT8_9.7'][:]
                bt9 = f['BT9_10.7'][:]
                bt10 = f['BT10_11.7'][:]
                Lat = f['BT_Lat'][:]
                Lon = f['BT_Lon'][:]

                nc_pac(Lon, Lat, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8, bt9,
                       bt10, os.path.join(input_directory, filename))
        except FileNotFoundError:
            print(f"No such file:{filename}")


