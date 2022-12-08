import numpy as np
from resources.utils.burn_level_utils import calculate_burn_level

def test_get_nbr():
    # Given
    # a filename
    path = "./mock_maps_LandSat8/Al8_100066_010919_001.tif"

    # Then
    burn_level_layers = calculate_burn_level(path, 1)

    # Then
    assert np.array(burn_level_layers).shape == (3,512,512)

if __name__ == '__main__':
    test_get_nbr()