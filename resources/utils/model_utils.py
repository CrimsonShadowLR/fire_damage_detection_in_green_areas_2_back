import torch

import resources.utils.models as models
from resources.utils.dataset import load_image
from resources.utils.image_utils import preprocess_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def image_loader(img, satelite):
    """
    Preprocesa una imagen para ser apta para entrar en el modelo de segmentación.
    """
    img = load_image(img)
    img = preprocess_image(img, satelite)
    return img

def predict(model, image, satellite_opt):
    """
    Procesa una imagen satelital, previamente preprocesada, mediante un modelo de
    segmentación y devuelve una
    máscara que señala los cuerpos de agua de la imagen original.
    """

    img_input = image_loader(image, satellite_opt)
    trained_model = model
    mask_predicted = run_model(img_input, trained_model)
    print("mask_predicted", mask_predicted.shape)
    del trained_model

    return mask_predicted.cpu().numpy()


def load_model(model_path, input_channels):
    """
    Carga en GPU un modelo de PyTorch.

    :param model_path: archivo .pth que representa al modelo
    :type model_path: str

    :param input_channels: bandas de la imagen de entrada
    :type input_channels: int

    :param num_classes: cantidad de clases de predicción
    :type num_classes: int

    :rtype: torch.nn.Module
    """
    model = models.UNet11(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def run_model(patch, model):
    """
    Ejecuta un modelo de PyTorch.

    :param patch: imagen a procesar;
    :type patch: torch.autograd.Variable

    :param model: modelo de PyTorch
    :type model: torch.nn.Module
    """
    model.eval()
    # print("Model in eval mode")
    with torch.set_grad_enabled(False):
        response = torch.sigmoid(model(patch))
    return response


def run_model_softmax(patch, model):
    """
    Ejecuta un modelo de PyTorch.

    :param patch: imagen a procesar;
    :type patch: torch.autograd.Variable

    :param model: modelo de PyTorch
    :type model: torch.nn.Module
    """
    model.eval()
    # print("Model in eval mode")
    with torch.set_grad_enabled(False):
        response = torch.exp(model(patch))
    return response
