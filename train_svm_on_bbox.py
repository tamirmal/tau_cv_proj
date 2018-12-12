from SVC_over_CNN import svc_over_cnn
from VGG_feature_extract import vgg_features_extract
import os
import numpy as np
import pickle


def main():
#    base_path = '/home/tamirmal/workspace/git/tau_cv_proj'
    base_path = '/home/tamir/git/tau_cv_proj'
    base_path_dataset = base_path + '/DATASET'

    img_list = []

    if not os.path.isdir(base_path):
        print("Cant locate directory {}".format(base_path))
        exit(-1)

    for cls in os.listdir(base_path_dataset):
        cls_path = base_path_dataset + '/' + cls
        if not os.path.isdir(cls_path):
            print("base_path should contain only directories, {} is not a directory".format(cls))
            exit(-1)

        for img in os.listdir(cls_path):
            img_path = cls_path + '/' + img
            img_suffix = img_path.split('.')[-1]
            if img_suffix.lower() != 'jpg':
                print("img path : {} not in jpg format!".format(img_path))
                exit(-1)

            img_list.append(
                {
                    'name': cls_path + '/' + img,
                    'class': cls,
                })

    if True:
        X, Y = vgg_features_extract.vgg_prepare_features_for_train(img_list)
        X = np.reshape(X, (len(X), -1))

    cls = svc_over_cnn.train_classifier(X, Y)
    with open(base_path + '/svm_best_cls.dump') as outf:
        pickle.dump(cls, outf)

# End of main


if __name__ == "__main__":
    main()
