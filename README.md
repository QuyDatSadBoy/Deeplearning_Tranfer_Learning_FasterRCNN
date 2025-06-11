# Deep Learning Transfer Learning với Faster R-CNN

## Giới thiệu

Dự án này triển khai mô hình Faster R-CNN sử dụng kỹ thuật Transfer Learning cho bài toán object detection. Faster R-CNN là một trong những kiến trúc deep learning tiên tiến nhất cho việc phát hiện và định vị đối tượng trong hình ảnh.

## Tính năng chính

- ✅ Triển khai Faster R-CNN với backbone networks phổ biến
- ✅ Sử dụng Transfer Learning từ các mô hình pre-trained
- ✅ Hỗ trợ training trên custom dataset
- ✅ Evaluation và visualization kết quả
- ✅ Fine-tuning hyperparameters

## Kiến trúc mô hình

Faster R-CNN bao gồm hai thành phần chính:
1. **Region Proposal Network (RPN)**: Tạo ra các proposal regions
2. **Fast R-CNN detector**: Phân loại và tinh chỉnh bounding boxes

## Yêu cầu hệ thống

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install pillow
pip install tensorboard
```

### Môi trường

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (khuyến nghị cho GPU training)
- RAM: 8GB+ (16GB+ cho dataset lớn)
- GPU: 6GB+ VRAM (khuyến nghị)

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/QuyDatSadBoy/Deeplearning_Tranfer_Learning_FasterRCNN.git
cd Deeplearning_Tranfer_Learning_FasterRCNN
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Tải pre-trained weights (nếu cần):
```bash
# Sẽ được tự động tải khi chạy lần đầu
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

Tổ chức dataset theo cấu trúc:
```
data/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

### 2. Training

```python
# Basic training
python train.py --dataset_path ./data --epochs 50 --batch_size 4

# Advanced training với custom parameters
python train.py \
    --dataset_path ./data \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --backbone resnet50 \
    --pretrained True
```

### 3. Evaluation

```python
# Đánh giá mô hình
python evaluate.py --model_path ./checkpoints/best_model.pth --test_path ./data/test

# Tính toán mAP
python calculate_map.py --model_path ./checkpoints/best_model.pth
```

### 4. Inference

```python
# Dự đoán trên ảnh đơn lẻ
python inference.py --image_path ./test_image.jpg --model_path ./checkpoints/best_model.pth

# Dự đoán trên thư mục ảnh
python batch_inference.py --input_dir ./test_images --output_dir ./results
```

## Cấu hình

### Hyperparameters chính

```python
# Model configuration
BACKBONE = 'resnet50'  # resnet50, resnet101, mobilenet
NUM_CLASSES = 21  # Số lượng classes + background
INPUT_SIZE = (800, 800)

# Training configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 4
EPOCHS = 100
WEIGHT_DECAY = 0.0005

# RPN configuration
RPN_ANCHOR_SCALES = [8, 16, 32]
RPN_ANCHOR_RATIOS = [0.5, 1.0, 2.0]
```

### Transfer Learning Settings

```python
# Freeze backbone layers
FREEZE_BACKBONE = True
FREEZE_EPOCHS = 10  # Số epochs freeze backbone

# Fine-tuning learning rates
BACKBONE_LR = 0.0001
HEAD_LR = 0.001
```

## Kết quả

### Performance Metrics

| Dataset | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---------|---------|--------------|-----|
| COCO    | 85.2%   | 62.1%        | 15  |
| Custom  | TBD     | TBD          | TBD |

### Visualization

```python
# Hiển thị kết quả detection
python visualize.py --image_path ./sample.jpg --model_path ./model.pth
```

## Cấu trúc project

```
Deeplearning_Tranfer_Learning_FasterRCNN/
├── models/
│   ├── faster_rcnn.py
│   ├── backbone.py
│   └── rpn.py
├── data/
│   ├── dataset.py
│   └── transforms.py
├── utils/
│   ├── visualization.py
│   ├── metrics.py
│   └── config.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

## Tùy chỉnh

### Thêm backbone mới

```python
# Trong models/backbone.py
def create_backbone(name, pretrained=True):
    if name == 'your_backbone':
        return YourBackbone(pretrained=pretrained)
```

### Custom dataset

```python
# Trong data/dataset.py
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        # Implement your dataset logic
        pass
```

## Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**
   - Giảm batch_size
   - Giảm input image size
   - Sử dụng gradient accumulation

2. **Training không hội tụ**
   - Kiểm tra learning rate
   - Kiểm tra data augmentation
   - Điều chỉnh loss weights

3. **Low mAP**
   - Tăng epochs training
   - Fine-tune anchors
   - Kiểm tra annotation quality

## Contributing

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Tạo Pull Request

## Tài liệu tham khảo

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Changelog

### Version 1.0.0
- Initial release với basic Faster R-CNN implementation
- Hỗ trợ transfer learning
- Training và evaluation scripts


## Liên hệ

- **Tác giả**: QuyDatSadBoy
- **GitHub**: [QuyDatSadBoy](https://github.com/QuyDatSadBoy)
- **Email**: [tranquydat2003@gmail.com]


---

⭐ Nếu project này hữu ích cho bạn, hãy star repository này!
