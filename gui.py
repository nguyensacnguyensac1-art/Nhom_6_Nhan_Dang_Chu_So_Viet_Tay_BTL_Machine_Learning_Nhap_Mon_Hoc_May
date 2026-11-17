# -*- coding: utf-8 -*-
"""Demo GUI - ĐỀ TÀI 5: NHẬN DẠNG CHỮ SỐ VIẾT TAY (ĐIỂM THƯỞNG).

Cách dùng:
  1. Chạy `python main.py` một lần để sinh mô hình models/gui_model.joblib
  2. Chạy `python gui.py`
  3. Vẽ một chữ số (0-9) vào ô vuông bằng chuột -> bấm [Dự đoán]
  4. Bấm [Xóa] để vẽ lại

Mô hình demo: K-NN (k=5, metric=euclidean) trên đặc trưng pixel 8x8.
Điểm mấu chốt: bạn VẼ trên canvas LỚN 28x28 (nét mịn, tự nhiên), khi bấm
[Dự đoán] ảnh được downsample về 8x8 (tạo giá trị chống răng cưa giống
load_digits) rồi đưa qua model. Model được train trên CÙNG pipeline này
(kèm dịch chuyển) nên nhận ra được cả chữ vẽ lệch vị trí/kích thước.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "gui_model.joblib"
CANVAS = 28      # lưới vẽ 28x28 (mịn hơn 8x8 để nét vẽ tự nhiên)
CELL_PX = 22     # kích thước mỗi ô trên canvas (pixel)


def build_gui():
    import joblib
    import numpy as np
    import tkinter as tk
    from tkinter import font as tkfont

    from src.features import downsample_canvas, preprocess_digit

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Chua co model: {MODEL_PATH}\nHay chay 'python main.py' truoc de sinh model demo."
        )
    pipe = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------------
    # Canvas vẽ 28x28 + lưu lưới
    # ------------------------------------------------------------------
    class DigitCanvas(object):
        def __init__(self, master):
            self.n = CANVAS
            self.cell = CELL_PX
            self.grid = np.zeros((self.n, self.n))
            self.canvas = tk.Canvas(
                master,
                width=self.n * self.cell,
                height=self.n * self.cell,
                bg="white",
                highlightthickness=1,
                highlightbackground="#999999",
            )
            self.canvas.bind("<Button-1>", self._paint)
            self.canvas.bind("<B1-Motion>", self._paint)
            self._redraw()

        def _paint(self, event):
            col, row = event.x // self.cell, event.y // self.cell
            if not (0 <= col < self.n and 0 <= row < self.n):
                return
            # Tô ô trỏ tới = 1.0, 4 ô lân cận (ngang/dọc) = 0.5
            # -> nét vẽ ~2 ô, đủ dày để nhìn rõ mà không "béo" quá.
            for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                r2, c2 = row + dy, col + dx
                if 0 <= r2 < self.n and 0 <= c2 < self.n:
                    v = 1.0 if (dx == 0 and dy == 0) else 0.5
                    self.grid[r2, c2] = max(self.grid[r2, c2], v)
            self._redraw()

        def _redraw(self):
            self.canvas.delete("all")
            for r in range(self.n):
                for c in range(self.n):
                    v = self.grid[r, c]
                    if v > 0.03:
                        shade = int(255 * (1 - v))
                        color = f"#{shade:02x}{shade:02x}{shade:02x}"
                        self.canvas.create_rectangle(
                            c * self.cell, r * self.cell,
                            (c + 1) * self.cell, (r + 1) * self.cell,
                            fill=color, outline="")
            # Lưới mờ
            for i in range(0, self.n + 1, 4):
                self.canvas.create_line(i * self.cell, 0, i * self.cell,
                                        self.n * self.cell, fill="#eeeeee")
                self.canvas.create_line(0, i * self.cell, self.n * self.cell,
                                        i * self.cell, fill="#eeeeee")

        def clear(self):
            self.grid = np.zeros((self.n, self.n))
            self._redraw()

        def array(self):
            return self.grid.copy()

    # ------------------------------------------------------------------
    # Cửa sổ chính
    # ------------------------------------------------------------------
    root = tk.Tk()
    root.title("Nhan dang chu so viet tay - De tai 5")
    root.resizable(False, False)

    title_font = tkfont.Font(size=13, weight="bold")
    label_font = tkfont.Font(size=11)

    tk.Label(root, text="Ve mot chu so (0-9) roi bam [Du doan]",
             font=title_font, fg="#2c3e50").pack(pady=(12, 4))

    canvas = DigitCanvas(root)
    canvas.canvas.pack()

    result_var = tk.StringVar(value="Du doan: --")
    prob_var = tk.StringVar(value="Xac suat: --")
    top_var = tk.StringVar(value="")

    tk.Label(root, textvariable=result_var, font=label_font,
             fg="#c0392b").pack(pady=(8, 0))
    tk.Label(root, textvariable=prob_var, font=label_font,
             fg="#2980b9").pack()
    tk.Label(root, textvariable=top_var, font=tkfont.Font(size=9),
             fg="#7f8c8d").pack()

    def predict():
        grid = canvas.array()  # (28,28) giá trị [0,1]
        if np.max(grid) < 0.03:
            result_var.set("Ban chua ve gi ca!")
            return

        # Pipeline KHỚP với cách train model:
        #   canvas 28x28 -> downsample 8x8 -> preprocess (crop + căn giữa)
        small = downsample_canvas(grid)
        small = preprocess_digit(small)

        x = small.reshape(1, -1)
        if pipe.get("scaler") is not None:
            x = pipe["scaler"].transform(x)
        model = pipe["model"]

        pred = int(model.predict(x)[0])
        result_var.set(f"Du doan: {pred}")
        try:
            proba = model.predict_proba(x)[0]
            prob_var.set(f"Xac suat: {100 * proba[pred]:.1f}%")
            top3 = np.argsort(proba)[::-1][:3]
            top_var.set("Top 3: " +
                        " | ".join(f"{int(t)}: {100 * proba[t]:.1f}%" for t in top3))
        except Exception:
            prob_var.set("Xac suat: --")

    def clear():
        canvas.clear()
        result_var.set("Du doan: --")
        prob_var.set("Xac suat: --")
        top_var.set("")

    buttons = tk.Frame(root)
    buttons.pack(pady=12)
    tk.Button(buttons, text="Du doan", command=predict, width=14,
              bg="#27ae60", fg="white", font=label_font).pack(side="left", padx=6)
    tk.Button(buttons, text="Xoa", command=clear, width=14,
              bg="#e74c3c", fg="white", font=label_font).pack(side="left", padx=6)

    root.mainloop()


if __name__ == "__main__":
    build_gui()