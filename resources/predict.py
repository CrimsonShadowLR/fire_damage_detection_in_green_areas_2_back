import os
import time
from copy import deepcopy

import rasterio
from flask import request
from flask_restful import Resource
from rasterio.io import MemoryFile
from resources.utils.burn_level_utils import calculate_burn_level
from resources.utils.model_utils import load_model, predict

import status
from resources.utils.image_utils import get_bounding_box, convert_mask_to_png, convert_raster_to_png
from resources.utils.json_utils import build_response

UPLOAD_DIRECTORY = "static/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


class PredictResource(Resource):
    def post(self):
        """Recurso RESTful que devuelve una máscara que señala los cuerpos de agua presentes en una imagen satelital
        de cuatro bandas. Este recurso requiere recibir, además de la imagen, el nombre de la misma para asignarle a
        la máscara uno derivado de esta.

        Argumentos:
        file:     imagen satelital de cuatro bandas y de cuatro bandas;
        filename: nombre de archivo de la imagen satelital
        """

        start = time.time()
        print("Receiving image...")
        request_dict = request.get_json()
        if not request_dict:
            response = {'error': 'No data provided'}
            print("No data provided!")
            return response, status.HTTP_400_BAD_REQUEST

        filepath = request_dict["filepath"]
        satellite_opt = request_dict["satellite"]

        # initialize parameters
        # LandSat8
        if satellite_opt==1:
            model_name="landsat8"
            satellite_image_folder="mock_maps_LandSat8/"
            n_channel=8
        # Sentinel2
        else:
            model_name="sentinel2"
            satellite_image_folder="mock_maps_Sentinel2/"
            n_channel=5

        if filepath is None:
            response = {'error': 'No filepath given'}
            print("No filepath given!")
            return response, status.HTTP_400_BAD_REQUEST

        last_slash = filepath.rfind("/") + 1  # Ocurrencia de la última diagonal + 1
        last_dot = filepath.rfind(".")  # Ocurrencia del último punto
        filename = filepath[last_slash:last_dot]  # Nombre de la imagen

        print("Filename: {}".format(filename))

        # Leyendo imagen
        file = open(filepath, "rb")
        data = file.read()
        reading = time.time()
        print(
            "Image Received. Elapsed time: {}s".format(str(round(reading - start, 2)))
        )
        print("Opening image...")

        memfile = MemoryFile(data)
        dataset = memfile.open()
        meta = dataset.profile
        img_npy = dataset.read()

        opening = time.time()
        print(
            "Image opened. Elapsed time: {}s".format(str(round(opening - reading, 2)))
        )

        print("Image shape: {}".format(str(img_npy.shape)))

        print("Generating mask...")
        
        MODEL_PATH = "trained_models/" + model_name

        model = load_model(model_path=MODEL_PATH, input_channels=n_channel)

        mask = predict(model, satellite_image_folder+filename + ".tif", satellite_opt)
        mask = mask[0]
        layers_paths = []

        # calcular gravedad de quemaduras
        burn_level_layers = calculate_burn_level(satellite_image_folder+filename, satellite_opt)

        layers_paths.append(convert_raster_to_png(filename=filename, raster=img_npy, metadata=meta, sat_opt=satellite_opt))
        i=0
        for idx, layer in enumerate(mask):
            layer_path = convert_mask_to_png(filename=filename, raster=layer, metadata=meta, burn_level_layers=burn_level_layers)
            layers_paths.append(layer_path)

        predicting = time.time()
        print(
            "Mask generated! Elapsed time: {}s".format(str(round(predicting - opening, 2)))
        )
        bounding_box = get_bounding_box(memfile.open())

        mask = mask.astype('uint8')
        metadata = deepcopy(meta)
        metadata['driver'] = "GTiff"
        metadata['count'] = mask.shape[0]
        metadata['height'] = mask.shape[1]
        metadata['width'] = mask.shape[2]
        metadata['dtype'] = mask.dtype

        with rasterio.open(os.path.join(UPLOAD_DIRECTORY, filename + ".tif"), 'w', **metadata) as outds:
            outds.write(mask)


        end = time.time()
        response = build_response(bounding_box, layers_paths)
        print("Total Elapsed Time: {}s".format(str(round(end - start, 2))))

        return response, status.HTTP_200_OK
