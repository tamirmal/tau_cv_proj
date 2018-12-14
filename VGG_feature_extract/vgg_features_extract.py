from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


def vgg_extract_features_img_array(img_array, model):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)

    return features


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


# v2 : with data augmentation
def vgg_prepare_features_for_train_v2(img_list, desired_length = None):
    """
    Input :
        img_list : list of dicts containing img name and label

    returns :
        features_list : each element a dict, with features and image name
    """
    assert img_list is not None

    if desired_length is None:
        desired_length = 10 * len(img_list)

    model = VGG16(weights='imagenet', include_top=False)

    # Extract features over original image set
    features_list = []
    classes_list = []
    for img in img_list:
        img_path = img['name']
        assert img['class'] is not None

        features = vgg_extract_features(img_path, model)

        features_list.append(features)
        classes_list.append(img['class'])

    print("Original dataset - extracted {} features".format(len(img_list)))
    print("Applying data-augmentation")

    # Aggregate images to X, Y lists
    X = []
    Y = []
    for img in img_list:
        x = image.load_img(img['name'], target_size=(224, 224))
        x = image.img_to_array(x)
        X.append(x)
        Y.append(img['class'])
    X = np.array(X)

    shift = 0.1
    datagen = ImageDataGenerator(rotation_range=45, width_shift_range=shift, height_shift_range=shift, horizontal_flip=True)
    datagen.fit(X)

    for k in range(desired_length):
        for X_batch, y_batch in datagen.flow(X, Y, batch_size=1):
            # Change to True to visualize image
            if False:
                print("class {}".format(y_batch[0]))
                pyplot.figure()  # figure starts from 1 ...
                pyplot.imshow(image.array_to_img(X_batch[0]))
                pyplot.show()

#            print("extracting features from augmentation idx {}".format(k))
            features = vgg_extract_features_img_array(X_batch[0], model)
            features_list.append(features)
            classes_list.append(y_batch[0])
            break

    print("Recieved a dataset with len {}, after augmentation expanded to length of {}".format(len(img_list), desired_length))
    return features_list, classes_list

# End of vgg_prepare_features_for_train
