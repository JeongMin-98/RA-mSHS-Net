import argparse
import cv2
import matplotlib.pyplot as plt
from lib.dataset.COCOformat import COCOEncoder, KeypointDB
from pycocotools.coco import COCO


def arg_parser():
    parser = argparse.ArgumentParser(description="labelme2COCO")
    parser.add_argument("--input_dir",
                        default='./test/',
                        help="input annotated your directory")
    parser.add_argument("--output_dir",
                        default='./test/',
                        help="output dataset directory"
                        )
    args = parser.parse_args()

    return args


def visualize_keypoints(image_path, keypoints, heatmap_shape):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)

    if heatmap_shape is None:
        heatmap_shape = [512, 407]
    ratio_w = 512 / heatmap_shape[0]
    ratio_h = 407 / heatmap_shape[1]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x * ratio_w, y * ratio_h, s=100, c='red', marker='x')
        plt.text(x * ratio_w, y * ratio_h, str(i), fontsize=12, color='yellow')
    plt.show()


def main():
    args = arg_parser()

    origin_test_json = 'data/coco/annotations/Foot_New_Doctor2_test.json'
    origin = KeypointDB(args, origin_test_json, is_load_coco=True)
    origin.load_coco_json()

    origin_coco = COCO(origin_test_json)

    infer_test_json = 'output/retrained/coco/pose_resnet_50/384x288_d256x3_adam_lr1e-3-RHPE-Foot-N3-Doctor-noflip-test/results/keypoints_Foot_New_Doctor2_results.json'
    infer = KeypointDB(args, infer_test_json, is_load_coco=True)
    infer.load_coco_json()

    # keypoint alignment
    image_id = []
    kps = []
    for db in infer.db:
        image_id.append(db['image_id'])
        kps_per_images = []
        for i in range(0, len(db['keypoints']), 3):
            kp = [db['keypoints'][i], db['keypoints'][i + 1]]
            kps_per_images.append(kp)
        kps.append(kps_per_images)
    imgs = origin_coco.loadImgs(image_id)

    # config file로 이미지 경로 불러오기
    for image_info, kp in zip(imgs, kps):
        visualize_keypoints(image_info['file_name'], kp, heatmap_shape=None)

    return


if __name__ == '__main__':
    main()
