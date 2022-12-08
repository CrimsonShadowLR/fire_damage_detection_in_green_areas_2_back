
import rasterio

def return_bands(img, satellite_opt):
    if satellite_opt==1:
        b_band = img.read(2).astype('f4')
        g_band = img.read(3).astype('f4')
        r_band = img.read(4).astype('f4')
        nir_band = img.read(5).astype('f4')
        swir_band = img.read(7).astype('f4')
    else:
        b_band = img.read(1).astype('f4')
        g_band = img.read(2).astype('f4')
        r_band = img.read(3).astype('f4')
        nir_band = img.read(4).astype('f4')
        swir_band = img.read(5).astype('f4')
    return (r_band, b_band, g_band, nir_band, swir_band)

def return_index_nbr(value, satellite):
    # Sentinel 2
    if satellite==1:
        q25, mean, q75=0.055,0.144,0.243

    # LandSat 8
    else:
        q25, mean, q75=-(240e-3),-(167e-3),-(116e-3)

    # magenta high severity
    if value < q25:
        return 255,0,255
    # dark orange medium severity
    elif value >= q25 and value < mean:
        return 	255, 140, 0
    # yellow, low severity
    elif value >= mean and value < q75:
        return 243, 255, 0
    # green unburned
    elif value >= q75:
        return 0, 255, 0
    # null, out of range
    else:
        return 255,0,255

def calculate_burn_level(filename, satellite_opt):
    img = rasterio.open(filename + ".tif")

    # obtener las bandas
    _, _, _, nir_band, swir_band = return_bands(img, satellite_opt)

    nbr = (nir_band - swir_band) / (nir_band + swir_band)
    
    x, y = nbr.shape

    coloured_nbr_red=nbr.copy()
    coloured_nbr_green=nbr.copy()
    coloured_nbr_blue=nbr.copy()

    for i in range(x):
        for j in range(y):
            r, g, b = return_index_nbr(nbr[i][j], satellite_opt)
            coloured_nbr_red[i][j]=r
            coloured_nbr_green[i][j]=g
            coloured_nbr_blue[i][j]=b

    return coloured_nbr_red, coloured_nbr_green, coloured_nbr_blue
