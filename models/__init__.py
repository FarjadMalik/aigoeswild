"""

"""
import config
from .resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152


def get_model():
    mdl = resnet_50()
    if config.model == "resnet18":
        mdl = resnet_18()
    if config.model == "resnet34":
        mdl = resnet_34()
    if config.model == "resnet101":
        mdl = resnet_101()
    if config.model == "resnet152":
        mdl = resnet_152()
    mdl.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    mdl.summary()
    return mdl
