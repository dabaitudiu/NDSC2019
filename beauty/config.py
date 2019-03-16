from models import VGG16

def model(model_name):
    if (model_name == "VGG16"):
        return VGG16
