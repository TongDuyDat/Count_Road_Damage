# Download Framework
Để thực hiện download framework sử dụng lệnh sau: 
```bash
git clone https://github.com/TongDuyDat/Count_Road_Damage.git
```
# Cài đặt môi trường
Sau khi download framework thực hiện lệnh sau để cài đặt môi trường cần thiết
```bash
cd Count_Road_Damage
pip install -r requirements.txt 
```
# Chuẩn bị dataset theo chuẩn yolo theo cấu trúc:
```bash
|+-Dataset
|  |+-images
|  |  |+-train
|  |  |   |+img_train_1.jpg
|  |  |   |+img_train_2.jpg
|  |  |   |+.....
|  |  |   |+img_train_n.jpg
|  |  |+-val
|  |  |   |+img_val_1.jpg
|  |  |   |+img_val_2.jpg
|  |  |   |+.....
|  |  |   |+img_val_n.jpg
|  |+-labels
|  |  |+-train
|  |  |   |+img_train_1.txt
|  |  |   |+img_train_2.txt
|  |  |   |+.....
|  |  |   |+img_train_n.txt
|  |  |+-val
|  |  |   |+img_val_1.txt
|  |  |   |+img_val_2.txt
|  |  |   |+.....
|  |  |   |+img_val_n.txt
```
# Chỉnh sửa file customdata.yaml
```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../Data # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)
names:
  0: emtry
  1: D00
  2: D10
  3: D20
  4: D40
```
# Huấn luyện mô hình
```bash
python yolov5/train.py --weights yolov5\fine_tuning1.pt --data customdata.yaml --img 640 batch-size 16 --epochs 100 
```
Bạn cũng có thể thay đổi file weights và file data theo ý muốn của bạn bằng cách trỏ đường dẫn tới file
```bash
python yolov5/train.py --weights your_weights.pt --data your_data.yaml --img 640 batch-size 16 --epochs 100 
```

# Thực hiện việc triển khai mô hình 
```bash
python track.py --yolo_weights yolov5\fine_tuning1.pt --source video1.mp4 --show-vid --save-vid
```

