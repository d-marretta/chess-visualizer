import json
import shutil
import os
import cv2


def get_normalized_corners(img_path, corners):
    im = cv2.imread(img_path)
    height, width, _ = im.shape
    top_left, top_right, bottom_right, bottom_left = corners['top_left'], corners['top_right'], corners['bottom_right'], corners['bottom_left']
    top_left = (top_left[0]/width, top_left[1]/height)
    top_right = (top_right[0]/width, top_right[1]/height)
    bottom_right = (bottom_right[0]/width, bottom_right[1]/height)
    bottom_left = (bottom_left[0]/width, bottom_left[1]/height)

    return top_left, top_right, bottom_right, bottom_left

def main():
    with open('./backup/annotations.json', mode='r', encoding='utf-8') as input:
        annotations = json.load(input)
    if not os.path.isdir('./datasets'):
        os.makedirs('./datasets/dataset/images/train')
        os.makedirs('./datasets/dataset/images/val')
        os.makedirs('./datasets/dataset/images/test')
        os.makedirs('./datasets/dataset/labels/train')
        os.makedirs('./datasets/dataset/labels/val')
        os.makedirs('./datasets/dataset/labels/test')


    train_ids = annotations['splits']['chessred2k']['train']['image_ids']
    val_ids = annotations['splits']['chessred2k']['val']['image_ids']
    test_ids = annotations['splits']['chessred2k']['test']['image_ids']

    corners_anns = annotations['annotations']['corners']
    imgs_anns = annotations['images']

    for corners_ann in corners_anns:
        img_id = corners_ann['image_id']
        corners = corners_ann['corners']
        for img in imgs_anns:
            if img['id'] == img_id:
                source_img_path = './backup/'+img['path']
                img_name = img['file_name']

        dest_path = ''
        if img_id in train_ids:
            dest_path = './datasets/dataset/images/train/' + img_name
        elif img_id in val_ids:
            dest_path = './datasets/dataset/images/val/' + img_name
        else:
            dest_path = './datasets/dataset/images/test/' + img_name

        shutil.copyfile(source_img_path, dest_path)
        label_file_path = f'{dest_path[:-4]}.txt'.replace('images','labels')
        with open(label_file_path, mode='w', encoding='utf-8') as label_file:
            top_left, top_right, bottom_right, bottom_left = get_normalized_corners(source_img_path, corners)
            label = f'0 {top_left[0]} {top_left[1]} {top_right[0]} {top_right[1]} {bottom_right[0]} {bottom_right[1]} {bottom_left[0]} {bottom_left[1]}\n'
            label_file.write(label)

if __name__ == '__main__':
    main()