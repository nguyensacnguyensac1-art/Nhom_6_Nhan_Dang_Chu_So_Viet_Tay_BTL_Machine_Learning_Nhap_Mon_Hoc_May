# Bài tập lớn FIT4013 – Đề tài 5: Nhận dạng chữ số viết tay

Bộ code hoàn chỉnh cho đề tài **"Nhận dạng chữ số viết tay"** (Nhóm đề tài Thị giác máy tính),
dùng bộ dữ liệu [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
của scikit-learn.

Quy trình triển khai đúng theo hướng dẫn trong file *"Hướng dẫn thực hiện bài tập lớn học phần FIT4013"*:

```
Data Source → EDA → Preprocessing → Feature/Representation → Model → Evaluation → Error Analysis → Conclusion
```

---

## 1. Nội dung & kết quả đạt được

| Bước | Nội dung | File đầu ra |
|------|----------|-------------|
| EDA | Phân phối lớp, ảnh mẫu, phân bố pixel, dữ liệu thiếu/trùng lặp | `figures/eda_*.png` |
| Tiền xử lý | Chuẩn hóa pixel về `[0,1]` | — |
| Đặc trưng | Vector pixel (baseline) + **HOG** (ảnh phóng to 32x32) | — |
| Mô hình | **K-NN** (thay đổi `k`, `metric`) và **Cây quyết định** (thay đổi `criterion`, `max_depth`) | — |
| Đánh giá | Accuracy, Precision, Recall, F1-macro, Confusion Matrix | `results/comparison_results.csv` |
| Phân tích lỗi | Các cặp chữ số hay nhầm lẫn, ảnh dự đoán sai | `figures/error_analysis.png` |
| Demo (điểm thưởng) | Chạy `python gui.py`, **vẽ chữ số bằng chuột → dự đoán** | — |
| Mô hình tốt nhất | Lưu pipeline (scaler + model) để tái sử dụng | `models/best_model.joblib` |

So với các mô hình đã học: K-NN cho kết quả tốt trên ảnh 8x8 (đặc thù: ảnh nhỏ, ít đặc trưng,
K-NN phù hợp vì dữ liệu đã đồng nhất về kích thước và không cần học "hình dạng" phức tạp).
Cây quyết định dễ overfit khi `max_depth` lớn — chi tiết phân tích nằm trong phần kết quả khi chạy.

---

## 2. Cấu trúc repository

```
nhan-dang-chu-viet-tay/
├── README.md          # Hướng dẫn này
├── requirements.txt   # Danh sách thư viện cần cài
├── AI_USAGE.md        # Khai báo sử dụng AI (theo yêu cầu của học phần)
├── main.py            # CHẠY TOÀN BỘ PIPELINE
├── gui.py             # Demo: vẽ chữ số bằng chuột và dự đoán (điểm thưởng GUI)
├── src/
│   ├── __init__.py
│   ├── data.py        # Nạp dữ liệu + EDA
│   ├── features.py    # Đặc trưng pixel & HOG
│   ├── models.py      # Huấn luyện K-NN / Decision Tree + độ đo đánh giá
│   └── evaluate.py    # Chạy thử nghiệm, confusion matrix, phân tích lỗi
├── figures/           # (tạo tự động) Biểu đồ kết quả
├── results/           # (tạo tự động) Bảng so sánh CSV
└── models/            # (tạo tự động) Mô hình đã huấn luyện
```

---

## 3. Cài đặt môi trường

> Yêu cầu: Python 3.9 trở lên. Toàn bộ thư viện chỉ cần `pip`, không cần GPU.
> Virtual environment `.venv` **đã được tạo sẵn trong thư mục project** (nếu bạn chạy trong WSL).

### VS Code + WSL (môi trường của bạn)

Trong terminal của VS Code (đảm bảo terminal đang ở chế độ **WSL**, không phải PowerShell Windows):

```bash
cd /mnt/c/Users/TungDN7/Desktop/nhan-dang-chu-viet-tay
source .venv/bin/activate
# Nếu .venv chưa có hoặc thiếu thư viện:
pip install -r requirements.txt
```

> **Mẹo:** Trong VS Code, bấm `Ctrl+Shift+P` → gõ **"Python: Select Interpreter"** → chọn
> `.venv/bin/python` (đường dẫn trong project) để VS Code dùng đúng môi trường này.

**GUI (gui.py) trên WSL:** cần thêm 1 bước (chạy 1 lần, cần mật khẩu sudo):
```bash
sudo apt-get install -y python3-tk
```
Sau đó `python gui.py` sẽ mở cửa sổ demo nhờ WSLg (Windows 11 / Windows 10 22H2+).

### Windows (cmd hoặc PowerShell)

```bat
cd C:\Users\TungDN7\Desktop\nhan-dang-chu-viet-tay
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd nhan-dang-chu-viet-tay
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Cách chạy

### 4.1 Chạy toàn bộ quy trình

```bash
python main.py
```

Kết quả:
- In ra bảng EDA, bảng **so sánh tất cả các phương án** (đã sắp theo F1 giảm dần), phân tích lỗi.
- Tạo `figures/`, `results/`, `models/` trong thư mục dự án.

> Mẹo: muốn lưu lại toàn bộ text để viết báo cáo:
> `python main.py | tee ket_qua.txt`

### 4.2 Chạy demo GUI (vẽ bằng chuột)

```bash
python gui.py
```

Cửa sổ hiện ra → vẽ một chữ số (0–9) vào ô vuông trắng → bấm **Dự đoán**.
Bấm **Xóa** để vẽ lại.

> **Cách vẽ để đoán đúng (quan trọng):**
> - Vẽ **gọn giữa ô**, không sát mép; nét **liền, tự nhiên** (đừng tô cục mịch).
> - Vẽ **đúng hình dạng** chữ (cong như thật, không vẽ bằng nét thẳng cứng).
> - Độ dày nét **~2–3 ô** là tốt nhất (cọ của GUI đã đủ dày).
>
> Model đạt **100% (200/200)** khi vẽ đúng hình dạng chuẩn, và **~91.5%** trên
> mô phỏng vẽ chuột thực tế (nét mỏng/dày + nghiêng ±10° + lệch ±4 ô + run tay).

> **Vì sao GUI đoán đúng dù bạn vẽ lệch?**
> Ảnh trong `load_digits` đã được cắt & căn giữa sẵn, nhưng chữ bạn vẽ thường
> lệch vị trí/kích thước, nét mỏng/dày hơn — K-NN tính khoảng cách pixel nên
> rất nhạy. Vì vậy model GUI được train trên **cùng pipeline** với `gui.py`:
> canvas 28x28 → downsample 8x8 → **preprocess (crop → resize 6x6 → căn giữa 8x8)**,
> kèm **augmentation mô phỏng đầy đủ cách vẽ chuột**:
>   - **Độ dày nét**: erode (mỏng) / dilate (dày) 1–2 lần
>   - **Xoay nhẹ**: ±4°, ±8° (vẽ nghiêng)
>   - **Co giãn**: 0.85× / 1.15× (vẽ to/nhỏ)
>   - **Dịch chuyển**: 9 vị trí (vẽ lệch)
>   - **Làm mềm**: Gaussian (anti-alias, giống nét `load_digits`)
>
> Tổng cộng **177.903 mẫu** sau augmentation (từ 1797 ảnh gốc).
> (Mô hình demo train trên toàn bộ dữ liệu, không dùng để báo cáo kết quả.)

---

## 5. Kiến thức cần nắm để bảo vệ

- `load_digits`: 1797 ảnh **8x8 pixel, ảnh xám (0–16)**, 10 lớp (chữ số 0–9), dữ liệu đã cân bằng.
- Vì sao chuẩn hóa pixel về `[0,1]`: đưa về cùng thang đo, bắt buộc với K-NN (tính khoảng cách).
- Vì sao **StandardScaler phải fit trên train**: tránh *data leakage* (model "nhìn trộm" tập test).
- K-NN: `k` càng nhỏ càng nhạy nhiễu (overfit), `k` lớn làm mịn ranh giới (underfit).
- Cây quyết định: `max_depth` lớn → overfit; `entropy` vs `gini` thường chênh lệch ít.
- HOG mô tả **hướng cạnh** của ảnh, thay cho việc dùng trực tiếp pixel.
- Vì sao không đánh giá chỉ bằng Accuracy → phải xem Confusion Matrix, F1-macro, phân tích lỗi.

---

## 6. Ghi chú nộp bài

- Nếu muốn thêm Notebook cho báo cáo, có thể mở `main.py` trong Jupyter bằng lệnh
  `jupyter nbconvert --to notebook --execute main.py` (tùy chọn, không bắt buộc).
- Dataset `load_digits` có sẵn trong scikit-learn → không cần tải, không cần đưa vào repo.
- Khai báo AI: xem `AI_USAGE.md`.
