# Phân Loại Tế Bào Ung Thư Cổ Tử Cung Sử Dụng Học Máy

Một pipeline học máy toàn diện để phân loại tế bào cổ tử cung từ ảnh Pap smear thành năm lớp: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, và Superficial-Intermediate.

## Tổng Quan

Dự án này kết hợp deep learning để trích xuất đặc trưng với các bộ phân loại học máy truyền thống để đạt độ chính xác cao trong phân loại tế bào cổ tử cung. Pipeline bao gồm tiền xử lý ảnh, trích xuất đặc trưng từ nhiều mô hình pre-trained, giảm chiều dữ liệu bằng PCA, và phân loại sử dụng NuSVC với kernel polynomial.

## Dữ Liệu

- **Dataset SIPaKMeD**: Tải từ [Kaggle](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
- **Tổng số ảnh**: 4.049 ảnh tế bào đơn lẻ
- **Số lớp**: 5 loại tế bào cổ tử cung
- **Định dạng ảnh**: PNG, độ phân giải 2048×1536

## Cấu Trúc Dự Án
```
├── main.py              # Huấn luyện và dự đoán trên tập test
├── tuning.py            # Tuning siêu tham số của SVM
├── statistic.ipynb      # Phân tích và thống kê dataset
└── README.md
```

## Mô Tả File

### `main.py`
**Mục đích**: Pipeline huấn luyện và dự đoán chính
- Nạp dataset SIPaKMeD
- Tiền xử lý ảnh (resize về 64×64)
- Trích xuất đặc trưng từ ResNetV2, DenseNet, EfficientNet
- Áp dụng MinMaxScaler → PCA (0.95) → StandardScaler
- Huấn luyện bộ phân loại NuSVC với các tham số đã tối ưu
- Đánh giá trên tập test
- Sinh ra Confusion Matrix và Classification Report

### `tuning.py`
**Mục đích**: Tuning siêu tham số cho SVM
- Thực hiện grid search hệ thống trên các tổ hợp tham số
- Kiểm tra các giá trị NU, gamma, degree, C khác nhau
- Sử dụng 5-fold cross-validation để đánh giá
- Lưu các cấu hình tốt nhất và kết quả vào JSON
- Hỗ trợ xử lý phân tán trên nhiều GPU

### `statistic.ipynb`
**Mục đích**: Khám phá và phân tích dataset
- Thống kê ảnh (kích thước, định dạng)
- Trực quan hóa ảnh mẫu từ từng lớp
- Thống kê theo từng lớp và biểu đồ

## Bắt Đầu Nhanh

### 1. Chuẩn Bị Dataset

1. Tải từ [Kaggle SIPaKMeD dataset](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
2. Giải nén và đặt trong thư mục dự án:
```
   ./data/SIPaKMeD/
```

### 2. Chạy Huấn Luyện và Dự Đoán
```bash
python main.py
```

Kết quả đầu ra:
- Trọng số mô hình lưu trong `./models/`
- Dự đoán và chỉ số trên tập test
- Trực quan hóa Confusion Matrix
- Classification Report với F1-score từng lớp

### 3. Chạy Tuning Siêu Tham Số
```bash
python tuning.py
```

Kết quả đầu ra:
- Kết quả grid search lưu trong `./results/tuning_results.jsonl`
- Các cấu hình tốt nhất được xếp hạng theo F1-score
- Thời gian thực thi và log sử dụng tài nguyên

### 4. Phân Tích Dataset
```bash
jupyter notebook statistic.ipynb
```

## Kết Quả

### Hiệu Suất (Tập Test)
- **Độ chính xác (Accuracy)**: 95.43%
- **F1-Score (macro average)**: 95.44%
- **F1-Score (weighted average)**: 95.43%

### Hiệu Suất Theo Từng Lớp
| Lớp | Precision | Recall | F1-Score |
|-----|-----------|--------|----------|
| Dyskeratotic | 0.9518 | 0.9693 | 0.9605 |
| Koilocytotic | 0.8982 | 0.9091 | 0.9036 |
| Metaplastic | 0.9419 | 0.9182 | 0.9299 |
| Parabasal | 0.9872 | 0.9809 | 0.9840 |
| Superficial-Intermediate | 0.9940 | 0.9940 | 0.9940 |

## Các Bước Trong Pipeline

Ảnh gốc → Resize về 64×64 → Trích xuất đặc trưng từ 3 mô hình (ResNetV2, DenseNet, EfficientNet) → Ghép các đặc trưng → MinMaxScaler [0, 1] → PCA (95% phương sai → ~800 đặc trưng) → StandardScaler (mean=0, std=1) → NuSVC Classifier (kernel poly) → Dự đoán + điểm tin cậy

## Cấu Hình

Sửa các siêu tham số trong code:
```python
# Xử lý ảnh
IMAGE_SIZE = 64

# Trích xuất đặc trưng
MODELS = ['ResNetV2', 'DenseNet', 'EfficientNet']

# Tiền xử lý
MINMAX_RANGE = (0, 1)
PCA_VARIANCE = 0.95
SCALER_TYPE = 'StandardScaler'

# Tham số SVM
SVM_TYPE = 'nu-SVC'
SVM_NU = 0.7
SVM_GAMMA = 0.1
SVM_DEGREE = 1
SVM_KERNEL = 'poly'
```

## Hạn Chế và Hướng Cải Thiện

### Hạn Chế Hiện Tại
1. **Nhầm lẫn giữa các lớp**: Dyskeratotic và Koilocytotic có hình thái giống nhau (F1: 0.9036)
2. **Yêu cầu GPU**: Trích xuất đặc trưng yêu cầu bộ nhớ GPU đáng kể
3. **Phạm vi tham số hạn chế**: Một số tham số chỉ được kiểm tra ở ít giá trị
4. **Dataset nhỏ**: 4.049 ảnh có thể hạn chế khả năng học của deep learning

### Cải Thiện Tương Lai
1. Sử dụng 10-fold CV với grid search rộng hơn để tuning tham số tốt hơn
2. Thử các bộ phân loại khác (XGBoost, LightGBM, Gradient Boosting)
3. Áp dụng data augmentation nâng cao (SMOTE, GAN) cho các lớp ít dữ liệu
4. Thử các kiến trúc deep learning khác để trích xuất đặc trưng
5. Sử dụng focal loss để nhấn mạnh các mẫu khó phân loại
6. Kết hợp nhiều mô hình để tăng độ tin cậy

## Sử Dụng Trên Kaggle

1. Tạo notebook Kaggle mới
2. Thêm repository này như một nguồn dữ liệu
3. Thêm dataset SIPaKMeD từ [Kaggle](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)
4. Chạy các cell theo thứ tự: `main.py` → `tuning.py` → `statistic.ipynb`
5. Xuất kết quả và các mô hình đã huấn luyện

## Trích Dẫn

Nếu sử dụng công trình này, vui lòng trích dẫn:
```bibtex
@dataset{SIPaKMeD,
  title={SIPaKMeD: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells},
  author={Plissiti, M. E. et al.},
  year={2018},
  url={https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed}
}
```

## Giấy Phép

[Chỉ định giấy phép của bạn tại đây - MIT, Apache 2.0, v.v.]

## Liên Hệ

Để đặt câu hỏi hoặc báo cáo vấn đề, vui lòng mở issue trên GitHub.

---

**Lưu Ý**: Dự án này dành cho mục đích giáo dục. Luôn tham khảo các chuyên gia y tế có trình độ để chẩn đoán và sàng lọc ung thư cổ tử cung thực tế.
