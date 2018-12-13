from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os


def vgg_extract_features(img_path, model):
    if not os.path.isfile(img_path):
        assert 0

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)

    return features
# End of vgg_extract_features


def vgg_prepare_features_for_train(img_list):
    """
    Input :
        img_list : list of dicts containing img name and label

    returns :
        features_list : each element a dict, with features and image name
    """
    assert img_list is not None

    model = VGG16(weights='imagenet', include_top=False)

    features_list = []
    classes_list = []
    for img in img_list:
        img_path = img['name']
        assert img['class'] is not None

        features = vgg_extract_features(img_path, model)

        features_list.append(features)
        classes_list.append(img['class'])

    import ipdb; ipdb.set_trace()
    return features_list, classes_list
# End of vgg_prepare_features_for_train
