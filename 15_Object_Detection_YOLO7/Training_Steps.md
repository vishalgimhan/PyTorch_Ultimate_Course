# 1. Get Yolo Project
git clone https://github.com/WongKinYiu/yolov7.git

# 2. Get Weights and Store in Yolov7 folder
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
# or
curl -L -o yolov7-e6e.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

# 3. Adapt cfg file
- Create copy of yolov7\cfg\training\yolov7.yaml
- Set number of classes

# 4. Adapt data file
Under data/masks.yaml

# 5. Get raw images and labels
Download from Kaggle
source: http://www.kaggle.com/datasets/andrewmvd/face-mask-detection

# 6. Data Prep
- Convert Pascal Voc labels to Yolo format
- SPlit data into subfolders

# 7. Perform Training
- cd yolov7
- python train.py --weights yolov7-e6e.pt --data "data/masks.yaml" --workers 1 --batch-size 4 --img 416 --cfg cfg/training/yolov7-masks.yaml --name yolov7 --epochs 5
- clear cache if training fails

# Detection
python detect.py --weights runs/train/yolov7/weights/best.pt --conf 0.4 --img-size 640 --source ./test/images/maksssksksss824.png