import os
import cv2

def get_cameras_most_frequent_size(dataset_path):
    images_paths_list = os.listdir(dataset_path)
    camera_sizes_map = {}
    num_imgs = len(images_paths_list)
    for i, img_path in enumerate(images_paths_list):
        cam_name = img_path[:3]
        img = cv2.imread(os.path.join(dataset_path, img_path))
        if img is None:
            print(f"Cannot read image {img_path}")
            continue
        h, w = img.shape[:2]
        size = (h, w)
        if cam_name not in camera_sizes_map:
            camera_sizes_map[cam_name] = dict()
        if size not in camera_sizes_map[cam_name]:
            camera_sizes_map[cam_name][size] = 1
        else:
            camera_sizes_map[cam_name][size] += 1
        
        print(f" Done {int((i + 1) / num_imgs * 100)} %", end='\r')
    
    camera_most_freq_size_map = {
        k : sorted(camera_sizes_map[k].items(), key=lambda item: item[1], reverse=True)[0][0]
            for k in camera_sizes_map
    }
    return camera_most_freq_size_map

def resize(img, new_size):
    h, w = img.shape[:2]
    if (w, h) == new_size:
        return img
    
    return cv2.resize(img, new_size)

def equalize(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(img_ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, img_ycrcb)
    cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img