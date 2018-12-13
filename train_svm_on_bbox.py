from SVC_over_CNN import svc_over_cnn
from VGG_feature_extract import vgg_features_extract
import os
import numpy as np
import pickle
from sklearn.externals import joblib
from optparse import OptionParser


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

    X, Y = vgg_features_extract.vgg_prepare_features_for_train(img_list)
    X = np.reshape(X, (len(X), -1))
    cls = svc_over_cnn.train_classifier_v2(X, Y)

    if os.path.exists(out_cls_path):
        joblib.dump(cls, out_cls_path)
    else:
        print("Cannot save trained svm model to {0}.".format(out_cls_path))

# End of main


if __name__ == "__main__":
    main()
