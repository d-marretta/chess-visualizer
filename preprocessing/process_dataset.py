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

def format_segmentation_dataset():
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

def get_normalized_bb_coords(img_path, bbox):
    im = cv2.imread(img_path)
    height, width, _ = im.shape
    x_top_left, y_top_left, width_bbox, height_bbox = bbox[0], bbox[1], bbox[2], bbox[3]
    x_center, y_center = x_top_left+width_bbox//2, y_top_left+height_bbox//2
    
    x_center = x_center/width
    y_center = y_center/height
    width_bbox = width_bbox/width
    height_bbox = height_bbox/height

    return x_center, y_center, width_bbox, height_bbox

    
def format_detection_dataset():
    with open('./backup/annotations.json', mode='r', encoding='utf-8') as input:
        annotations = json.load(input)
    if not os.path.isdir('./datasets/detection_dataset'):
        os.makedirs('./datasets/detection_dataset/images/train')
        os.makedirs('./datasets/detection_dataset/images/val')
        os.makedirs('./datasets/detection_dataset/images/test')
        os.makedirs('./datasets/detection_dataset/labels/train')
        os.makedirs('./datasets/detection_dataset/labels/val')
        os.makedirs('./datasets/detection_dataset/labels/test')


    train_ids = annotations['splits']['chessred2k']['train']['image_ids']
    val_ids = annotations['splits']['chessred2k']['val']['image_ids']
    test_ids = annotations['splits']['chessred2k']['test']['image_ids']

    pieces_anns = annotations['annotations']['pieces']
    imgs_anns = annotations['images']

    for img in imgs_anns:
        if not ((img['id'] in train_ids) or (img['id'] in val_ids) or (img['id'] in test_ids)):
            continue

        label = ''
        for piece_ann in pieces_anns:
            img_id_piece = piece_ann['image_id']
            if img['id'] == img_id_piece:
                source_img_path = './backup/'+img['path']
                img_name = img['file_name']
                bb_coords = piece_ann['bbox']
                cat = piece_ann['category_id']
                x_center, y_center, width_bbox, height_bbox = get_normalized_bb_coords(source_img_path, bb_coords)
                label += f'{cat} {x_center} {y_center} {width_bbox} {height_bbox}\n'
        
        dest_path = ''
        if img['id'] in train_ids:
            dest_path = './datasets/detection_dataset/images/train/' + img_name
        elif img['id'] in val_ids:
            dest_path = './datasets/detection_dataset/images/val/' + img_name
        else:
            dest_path = './datasets/detection_dataset/images/test/' + img_name

        shutil.copyfile(source_img_path, dest_path)
        label_file_path = f'{dest_path[:-4]}.txt'.replace('images','labels')
        with open(label_file_path, mode='w', encoding='utf-8') as label_file:
            label_file.write(label)

if __name__ == '__main__':
    #format_segmentation_dataset()
    format_detection_dataset()