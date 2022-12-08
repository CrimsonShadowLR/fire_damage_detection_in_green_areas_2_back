from io import BytesIO
import glob

import pysftp
from flask import request
from flask_restful import Resource

import numpy as np

import status
from resources.utils.image_utils import get_bounding_box_from_file, rect_overlap


class SearchImagesResource(Resource):
    def post(self):
        request_dict = request.get_json()
        if not request_dict:
            error = {'error': 'No data provided.'}
            return error, status.HTTP_400_BAD_REQUEST

        # Los datos de fechas llegaran como fechaini, fechafin ->
        # Los datos de coordenadas llegaran como Norte, Oeste, Sur, Este
        # Top, left, bottom, right

        left = request_dict["left"]
        bottom = request_dict["bottom"]
        right = request_dict["right"]
        top = request_dict["top"]
        satellite_opt = request_dict["satellite"]

        area_of_interest = {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
        }

        # initialize parameters
        # LandSat8
        if satellite_opt==1:
            satellite_image_folder="./mock_maps_LandSat8"
        # Sentinel2
        else:
            satellite_image_folder="./mock_maps_Sentinel2"


        response = {
            "images": []
        }

        
        data_path = satellite_image_folder
        input_files = np.array(sorted(glob.glob(data_path + "/*.tif")))
        print(area_of_interest)
        print("-------------------------------------")

        for file in input_files:
            bounding_box = get_bounding_box_from_file(file)
            print(bounding_box)

            if rect_overlap(
                (bounding_box["top"], bounding_box["right"]),
                (bounding_box["bottom"], bounding_box["left"]),
                (area_of_interest["top"], area_of_interest["right"]),
                (area_of_interest["bottom"], area_of_interest["left"]),
            ):
                last_slash = file.rfind("/") + 1  # Ocurrencia de la última diagonal + 1
                last_dot = file.rfind(".")  # Ocurrencia del último punto
                filename = file[last_slash:last_dot]  # Nombre de la imagen
                response["images"].append({
                    "path": file,
                    "name": filename,
                    "bounding_box": bounding_box
                })

        return response, status.HTTP_200_OK

