from sklearn.externals import joblib
from optparse import OptionParser
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from VGG_feature_extract import vgg_features_extract
from matplotlib import pyplot
import matplotlib.image as mpimg


def main():
    parser = OptionParser()
    parser.add_option("-d", "--test_data", dest="dataset_dir", help="Path to test data.")
    parser.add_option("-c", "--clsf", dest="clsf", help="Path to stored classifier")
    (options, args) = parser.parse_args()

    test_path = options.dataset_dir

    print("Use test data : {}".format(test_path))
    print("Use classifier : {}".format(options.clsf))

    if not os.path.isfile(options.clsf):
        print("Cant locate {}".format(options.clsf))
        assert 0

    clf = joblib.load(options.clsf)
    model = VGG16(weights='imagenet', include_top=False)

    X = []
    IMG_LIST = []
    for img in os.listdir(test_path):
        IMG_LIST.append(img)
        IMG_PATH = test_path + '/' + img
        features = vgg_features_extract.vgg_extract_features(IMG_PATH, model)
        X.append(features)

    #import ipdb; ipdb.set_trace()
    X = np.array(X)
    X = np.reshape(X, (len(X), -1))
    Y = clf.predict(X)

    if True:
        for idx, img in enumerate(IMG_LIST):
            print("image {} predicted class {}".format(img, Y[idx]))
            IMG_PATH = test_path + '/' + img
            x = image.load_img(IMG_PATH, target_size=(224, 224))
            pyplot.figure()  # figure starts from 1 ...
            pyplot.imshow(image.array_to_img(x))
            pyplot.show()

    print("DONE")


if __name__ == "__main__":
    main()
