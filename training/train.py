import ultralytics as ul
import os

model = ul.YOLO('./models/yolo11n-seg.pt')

model.train(data='./training/dataset.yaml', project='chessboard_segment', epochs=50, imgsz=640, patience=15, device=0, batch=32)
os.rename('./chessboard_segment/train/weight/best.pt', './models/yolo11n-seg-best.pt')
model.val(data='./training/dataset.yaml', split='test', project='chessboard_segment', name='test')
