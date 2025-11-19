from ultralytics import YOLO

# 既存のモデルをロード（事前学習済みのYOLOv8モデル）
model = YOLO('yolov8s.pt')  # "yolov8.pt" 

# 追加学習を実施
results = model.train(
    data=r'/Users/ikomiki/workspace-outer/YOLO-flower/data.yaml',  # データセットの設定ファイル
    epochs=10,  # 学習回数（必要に応じて変更）
    batch=8,  # バッチサイズ
    imgsz=640,  # 画像サイズ（入力画像をリサイズ）
    device='mps'  # CPU 又は GPU（CUDA）
)