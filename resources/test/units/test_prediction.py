from resources.utils.model_utils import load_model, predict


def test_get_prediction():
    # Given
    # a filename
    path = "./mock_maps_LandSat8/Al8_100066_010919_001.tif"

    # the model
    MODEL_PATH = "trained_models/landsat8"

    model = load_model(model_path=MODEL_PATH, input_channels=8)

    # When
    mask = predict(model, path, 1)

    # Then
    assert mask.shape == (1,1,512,512)

if __name__ == '__main__':
    test_get_prediction()