from resources.utils.image_utils import get_bounding_box_from_name
from rasterio.io import MemoryFile


def test_get_coordenates():
    # Given
    # a filename
    path = "./mock_maps_LandSat8/Al8_100066_010919_001.tif"

    # the coordinates
    coordinates = {'left': 140.714, 'bottom': -8.884, 'right': 140.842, 'top': -8.756}

    # When
    response = get_bounding_box_from_name(path)

    # Then
    assert coordinates["left"] == response["left"]
    assert coordinates["bottom"] == response["bottom"]
    assert coordinates["right"] == response["right"]
    assert coordinates["top"] == response["top"]

if __name__ == '__main__':
    test_get_coordenates()