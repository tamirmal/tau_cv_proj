import SVC_over_CNN
import VGG_feature_extract
import os


def main():
    base_path = '/home/tamir/PycharmProjects/tau_proj_prep/CROPPED'

    img_list = []

    if not os.path.isdir(base_path):
        print("Cant locate directory {}".format(base_path))
        exit(-1)

    for cls in os.listdir(base_path):
        cls_path = base_path + '/' + cls
        if not os.path.isdir(cls_path):
            print("base_path should contain only directories, {} is not a directory".format(cls))
            exit(-1)

        img_path = cls_path + '/' + img
        img_suffix = img_path.split('.')[-1]
        if img_suffix.lower() != '.jpg':
            print("img path : {} not in jpg format!".format(img_path))
            exit(-1)

        for img in os.listdir(cls_path):
            img_list.append(
                {
                    'name': cls_path + '/' + img,
                    'class': cls,
                })

    X, Y = VGG_feature_extract.vgg_prepare_features_for_train(img_list)
    cls = SVC_over_CNN.train_classifier(X, Y)

# End of main


if __name__ == "__main__":
    main()
