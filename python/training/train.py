import ultralytics as ul
import os

def train_seg():
    model = ul.YOLO('./models/yolo11n-seg.pt')

    model.train(data='./training/seg_dataset.yaml', project='chessboard_segment', epochs=40, imgsz=640, patience=5, device=0, batch=32)
    os.rename('./chessboard_segment/train/weights/best.pt', './models/yolo11n-seg-best.pt')

    model = ul.YOLO('./models/yolo11n-seg-best.pt')
    model.val(data='./training/seg_dataset.yaml', split='test', project='chessboard_segment', name='test')

def train_detect():
    model = ul.YOLO('./models/yolo11n.pt')

    model.train(data='./training/detect_dataset.yaml', project='detect_pieces', epochs=40, imgsz=640, patience=5, device=0, batch=32)
    os.rename('./detect_pieces/train/weights/best.pt', './models/yolo11n-best.pt')

    model = ul.YOLO('./models/yolo11n-best.pt')
    model.val(data='./training/detect_dataset.yaml', split='test', project='detect_pieces', name='test')

if __name__ == '__main__':
    #train_seg()
    train_detect()
