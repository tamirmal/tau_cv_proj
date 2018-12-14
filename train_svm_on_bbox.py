from SVC_over_CNN import svc_over_cnn
from VGG_feature_extract import vgg_features_extract
import os
import numpy as np
import pickle
from sklearn.externals import joblib
from optparse import OptionParser
from sklearn.model_selection import train_test_split


def main():
    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset_dir", help="Path to base dataset.")
    parser.add_option("-c", "--clsf", dest="clsf_out",
        help="Path to store trained classifier, will be stored as svm_best_cls.dump")
    (options, args) = parser.parse_args()

    base_path_dataset = options.dataset_dir

    out_cls_path = options.clsf_out + '/svm_best_cls.dump'
    if not os.path.isfile(out_cls_path):
        print("need to first create classifier output file : {}".format(out_cls_path))
        assert 0

    img_list = []
    class_list = []

    if not os.path.isdir(base_path_dataset):
        print("Cant locate directory {}".format(base_path_dataset))
        assert 0

    for cls in os.listdir(base_path_dataset):
        cls_path = base_path_dataset + '/' + cls
        if not os.path.isdir(cls_path):
            print("base_path should contain only directories, {} is not a directory".format(cls))
            assert 0

        for img in os.listdir(cls_path):
            img_path = cls_path + '/' + img
            img_suffix = img_path.split('.')[-1]
            if img_suffix.lower() != 'jpg':
                print("img path : {} not in jpg format!".format(img_path))
                assert 0

            img_list.append(
                {
                    'name': cls_path + '/' + img,
                    'class': cls,
                })
            class_list.append(cls)

    # Split data to train & test
    img_list_train, img_list_test, class_list_train, class_list_test = train_test_split(img_list, class_list, test_size=0.2)
    # Extract features + perform augmentation
    X_train, Y_train = vgg_features_extract.vgg_prepare_features_for_train_v2(img_list_train, class_list_train)
    X_test, Y_test = vgg_features_extract.vgg_prepare_features_for_train_v2(img_list_test, class_list_test)
    X_train = np.reshape(X_train, (len(X_train), -1))
    X_test = np.reshape(X_test, (len(X_test), -1))
    # Feed to SVM fitting, will test result with the test set
    cls = svc_over_cnn.train_classifier_v2(X_train, X_test, Y_train, Y_test)

    if os.path.exists(out_cls_path):
        joblib.dump(cls, out_cls_path)
    else:
        print("Cannot save trained svm model to {0}.".format(out_cls_path))

# End of main


if __name__ == "__main__":
    main()
