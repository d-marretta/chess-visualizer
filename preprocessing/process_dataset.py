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

def format_segmentation_real():
    with open('./backup/annotations.json', mode='r', encoding='utf-8') as input:
        annotations = json.load(input)
    if not os.path.isdir('./datasets/segmentation_dataset'):
        os.makedirs('./datasets/segmentation_dataset/images/train')
        os.makedirs('./datasets/segmentation_dataset/images/val')
        os.makedirs('./datasets/segmentation_dataset/images/test')
        os.makedirs('./datasets/segmentation_dataset/labels/train')
        os.makedirs('./datasets/segmentation_dataset/labels/val')
        os.makedirs('./datasets/segmentation_dataset/labels/test')


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
            dest_path = './datasets/segmentation_dataset/images/train/' + img_name
        elif img_id in val_ids:
            dest_path = './datasets/segmentation_dataset/images/val/' + img_name
        else:
            dest_path = './datasets/segmentation_dataset/images/test/' + img_name

        shutil.copyfile(source_img_path, dest_path)
        label_file_path = f'{dest_path[:-4]}.txt'.replace('images','labels')
        with open(label_file_path, mode='w', encoding='utf-8') as label_file:
            top_left, top_right, bottom_right, bottom_left = get_normalized_corners(source_img_path, corners)
            label = f'0 {top_left[0]} {top_left[1]} {top_right[0]} {top_right[1]} {bottom_right[0]} {bottom_right[1]} {bottom_left[0]} {bottom_left[1]}\n'
            label_file.write(label)

def format_segmentation_rendered(max_train_imgs, max_val_imgs, max_test_imgs):
    counter = 0

    for file in os.listdir('ChessRender360/annotations/'):
        if counter > max_train_imgs+max_val_imgs+max_test_imgs:
            break
        with open(f'./ChessRender360/annotations/{file}', mode='r', encoding='utf-8') as file_ann:
            ann = json.load(file_ann)

        img_path = f'./{ann['rgb_path']}'
        img = cv2.imread(img_path)

        height, width, _ = img.shape
        corner1 = (ann['board_corners']['white_left'][0],ann['board_corners']['white_left'][1])
        corner2 = (ann['board_corners']['white_right'][0],ann['board_corners']['white_right'][1])
        corner3 = (ann['board_corners']['black_left'][0],ann['board_corners']['black_left'][1])
        corner4 = (ann['board_corners']['black_right'][0],ann['board_corners']['black_right'][1])
        corners = [(corner1[0]/width, corner1[1]/height), (corner2[0]/width, corner2[1]/height), (corner3[0]/width, corner3[1]/height), (corner4[0]/width, corner4[1]/height)]

        split = ''
        if counter < max_train_imgs:
            split = 'train'
        elif counter >= max_train_imgs and counter < max_train_imgs + max_val_imgs:
            split = 'val'
        else:
            split = 'test'
        
        dataset_imgs_path = f'./datasets/segmentation_dataset_new/images/{split}/'
        dataset_labels_path = f'./datasets/segmentation_dataset_new/labels/{split}/'
        img_name = img_path.split('/')[3]
        shutil.copyfile(img_path, f'{dataset_imgs_path}{img_name}')

        with open(f'{dataset_labels_path}{img_name.split('.')[0]}.txt', mode='w', encoding='utf-8') as label_file:
            corner1, corner2, corner3, corner4 = corners[0], corners[1], corners[2], corners[3]
            label = f'0 {corner1[0]} {corner1[1]} {corner2[0]} {corner2[1]} {corner3[0]} {corner3[1]} {corner4[0]} {corner4[1]}'
            label_file.write(label)

        counter += 1

def format_segmentation_rendered2(train_num, test_num, val_num):
    counter = 0
    for file in os.listdir('./data'):
        if counter > train_num+test_num+val_num:
            break
        ext = file.split('.')[-1]
        if ext == 'json' and file.split('.')[0] != 'config':
            with open(f'./data/{file}', mode='r', encoding='utf-8') as input_json:
                ann = json.load(input_json)
            corner1 = (ann['corners'][0][0], 1-ann['corners'][0][1])
            corner2 = (ann['corners'][1][0], 1-ann['corners'][1][1])
            corner3 = (ann['corners'][2][0], 1-ann['corners'][2][1])
            corner4 = (ann['corners'][3][0], 1-ann['corners'][3][1])
            img_path = f'./data/{file.split('.')[0]}.jpg'
            split = ''
            if counter < train_num:
                split = 'train'
            elif counter >= train_num and counter < train_num + val_num:
                split = 'val'
            else:
                split = 'test'
                
            dataset_imgs_path = f'./datasets/segmentation_dataset_new/images/{split}/'
            dataset_labels_path = f'./datasets/segmentation_dataset_new/labels/{split}/'
            img_name = img_path.split('/')[-1]
            shutil.copyfile(img_path, f'{dataset_imgs_path}{img_name}')
            with open(f'{dataset_labels_path}{img_name.split('.')[0]}.txt', mode='w', encoding='utf-8') as label_file:
                label = f'0 {corner1[0]} {corner1[1]} {corner2[0]} {corner2[1]} {corner3[0]} {corner3[1]} {corner4[0]} {corner4[1]}'
                label_file.write(label)

            counter += 1


def move_old_to_new_dataset(train_num, test_num, val_num):
    for split, max_num in [('train', train_num), ('test',test_num), ('val',val_num)]:
        counter = 0
        for file in os.listdir(f'./datasets/segmentation_dataset/images/{split}/'):
            if counter > max_num:
                break
            shutil.copyfile(f'./datasets/segmentation_dataset/images/{split}/{file}', f'./datasets/segmentation_dataset_new/images/{split}/{file}')
            shutil.copyfile(f'./datasets/segmentation_dataset/labels/{split}/{file.split('.')[0]}.txt', f'./datasets/segmentation_dataset_new/labels/{split}/{file.split('.')[0]}.txt')
            counter += 1

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
    #format_segmentation_real()
    #format_detection_dataset()
    format_segmentation_rendered(3000, 450, 450)
    move_old_to_new_dataset(750,112,112)
    format_segmentation_rendered2(1555,193,193)