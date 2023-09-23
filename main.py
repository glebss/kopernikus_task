import os
import shutil
import argparse
import cv2
from collections import OrderedDict

from imaging_interview import preprocess_image_change_detection,\
                              compare_frames_change_detection
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the dataset to be processed")
    parser.add_argument("out_folder", type=str, help="Path to save resulting unique images")
    parser.add_argument("--blur-radius-list", type=str, help="Comma separated list of gausian blur radii to apply to an image")
    parser.add_argument("--min-contour-area", type=int, default=36, help="Minimum contour area on the diff image to take into account")
    parser.add_argument("--threshold-score", type=float, default=0.1, help="Minimum relative area in order to consider images different")
    parser.add_argument("--calc-features-map-size", type=int, default=200, help="Maximum number of preprocessed images to save"
                                                                                 "in order to not recalculate it each tiem")
    parser.add_argument("--equalize", '-e', action="store_true", help="Use histogram equalization")
    parser.add_argument("--remove-origs", action="store_true", help="Remove original images")
    args = parser.parse_args()
    if args.blur_radius_list:
        args.blur_radius_list = list(map(int, args.blur_radius_list.split(',')))
    if args.threshold_score:
        assert args.threshold_score >= 0 and args.threshold_score <= 1, "threshold-score must be in the [0, 1] range"
    return args

def main():
    args = parse_args()

    print("Getting cameras resolutions...")
    camera_most_freq_size_map = utils.get_cameras_most_frequent_size(args.dataset)
    images_to_save = []
    images_to_save_preprocessed = OrderedDict()

    print("Processing...")
    images_paths_list = os.listdir(args.dataset)
    num_imgs = len(images_paths_list)
    for i, img_path in enumerate(images_paths_list):
        save = True
        img = cv2.imread(os.path.join(args.dataset, img_path))
        if img is None:
            print(f"Cannot read image {img_path}")
            continue
        cam_name = img_path[:3]
        new_size = camera_most_freq_size_map[cam_name][::-1]
        img = utils.resize(img, new_size)
        if args.equalize:
            img = utils.equalize(img)
        img_preproc = preprocess_image_change_detection(img, gaussian_blur_radius_list=args.blur_radius_list)
        for img_path_to_save in images_to_save:
            cam_name_saved = img_path_to_save[:3]
            if cam_name_saved != cam_name:
                continue
            
            if img_path_to_save in images_to_save_preprocessed:
                img_saved_preproc = images_to_save_preprocessed[img_path_to_save]
            else:
                img_saved = cv2.imread(os.path.join(args.dataset, img_path_to_save))
                img_saved = utils.resize(img_saved, new_size)
                if args.equalize:
                    img_saved = utils.equalize(img_saved)
                img_saved_preproc = preprocess_image_change_detection(img_saved, gaussian_blur_radius_list=args.blur_radius_list)
            
            score, *_ = compare_frames_change_detection(img_preproc, img_saved_preproc, args.min_contour_area)
            rel_score = score / img_saved_preproc.size
            if rel_score < args.threshold_score:
                save = False
                break
        if save:
            images_to_save.append(img_path)
            images_to_save_preprocessed[img_path] = img_preproc
            if len(images_to_save_preprocessed) > args.calc_features_map_size:
                images_to_save_preprocessed.popitem(last=False)

        print(f" Done {int((i + 1) / num_imgs * 100)} %", end='\r')

    
    print(f"Saving {len(images_to_save)} resulting images to {args.out_folder}...")
    os.makedirs(args.out_folder, exist_ok=True)
    for img_path in images_to_save:
        src = os.path.join(args.dataset, img_path)
        dst = os.path.join(args.out_folder, img_path)
        shutil.copy(src, dst)

    if args.remove_origs:
        shutil.rmtree()

if __name__ == "__main__":
    main()
