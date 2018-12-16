import os
from faster_rcnn import predict_faster_rcnn
from predict_svm_bbox import predict_classes
from sklearn.externals import joblib
from keras.applications.vgg16 import VGG16
import cv2
from keras.preprocessing import image


def visualize_bboxes(name, img, bboxes, y, class_to_color):
    key = 'Bus'
    all_dets = []

    for cls, box in zip(y, bboxes):
        x1, y1 = box['x_min'], box['y_min']
        x2, y2 = box['x_max'], box['y_max']
        prob = box['prob']

        cv2.rectangle(img, (x1, y1), (x2, y2),
                      (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                      thickness=10)

        all_dets.append((key, 100 * prob))

        textLabel = '{}: {}'.format(cls, int(100 * prob))
        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        textOrg = (x1, y1 - 0)

        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


    from matplotlib import pyplot as plt
    print("{} : ".format(name) + str(all_dets))
    plt.figure()  # figure starts from 1 ...
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def run_wrapped_model(out_annotations_file, img_base_path, visualize = False):
    if os.path.isfile(out_annotations_file):
        print("output annotations file {} already exists".format(out_annotations_file))
        assert 0

    # Load Faster-RCNN models
    this_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(this_path, 'faster_rcnn/config.pickle')
    model_path = os.path.join(this_path, 'faster_rcnn/model_frcnn.hdf5')

    # Load SVM classifier
    svm_path = os.path.join(this_path, 'svm_best_cls.dump')
    clf = joblib.load(svm_path)
    svm_vgg = VGG16(weights='imagenet', include_top=False)

    # Collect image names
    img_names = [
        f for f in os.listdir(img_base_path) if f.lower().endswith(('.jpeg', '.jpg'))
    ]

    regions_preds = predict_faster_rcnn.predit_images(img_names, img_base_path, config_path, model_path, visualize)

    out_anns_list = []
    # Create warped images for classification
    for img, values in regions_preds.items():
        if len(values['bboxes']) == 0:
            continue

        results = {
            'name': img,
            'bboxes': None,
        }

        orig_img = values['image']
        img_arr_to_predict = []
        for region in values['bboxes']:
            x1 = region['x_min']
            x2 = region['x_max']
            y1 = region['y_min']
            y2 = region['y_max']
            warped_img_region = orig_img[y1:y2, x1:x2]
            warped_img_region = cv2.resize(warped_img_region, (224,224))

            if visualize:
                from matplotlib import pyplot
                pyplot.figure()
                pyplot.imshow(warped_img_region)
                pyplot.show()

            warped_img_region = image.img_to_array(warped_img_region)
            img_arr_to_predict.append(warped_img_region)

        Y = predict_classes(img_arr_to_predict, clf, svm_vgg, visualize)

        all_regions_str = ""
        for cls_res, region in zip(Y, values['bboxes']):
            x1 = region['x_min']
            x2 = region['x_max']
            y1 = region['y_min']
            y2 = region['y_max']
            w = x2 - x1
            h = y2 - y1
            all_regions_str += "[{},{},{},{},{}]".format(x1, y1, w, h, cls_res)

        img_str = "{}:" + all_regions_str

    print("DONE")

# End

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
    parser.add_option("-o", "--out", dest="out", help="out path.")
    (options, args) = parser.parse_args()

    run_wrapped_model(options.test_path, options.out, True)


if __name__ == "__main__":
    main()
