"""

"""
# import keras
# import tensorflow as tf
# from classification_models.keras import Classifiers

import config
from .resnet import resnet_18, resnet_50, resnet_34, resnet_152, resnet_101


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


# def load_pretrained_imagenet_resnet18_model():
#     # resnet18 model and _ is the preprocessed inputs
#     resnet18, _ = Classifiers.get('resnet18')
#     base_model = resnet18(input_shape=(config.image_height, config.image_width, config.channels),
#                           weights='imagenet', include_top=False)
#     x = keras.layers.GlobalAveragePooling2D()(base_model.output)
#     output = keras.layers.Dense(config.NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
#     model = keras.models.Model(inputs=[base_model.input], outputs=[output])
#     model.summary()
#     return model
