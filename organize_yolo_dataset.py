import os
import shutil
import random
from pathlib import Path

# 設定
FLOWERS_DIR = './flower_photos'
TRAIN_FRACTION = 0.8
RANDOM_SEED = 2018

# クラス名のマッピング（ディレクトリ名 -> クラスID）
CLASS_MAPPING = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}

def organize_yolo_dataset():
    """flower_photosディレクトリの内容をYOLOの学習形式に変換"""
    
    # ランダムシードを設定
    random.seed(RANDOM_SEED)
    
    # 出力ディレクトリを作成
    train_images_dir = Path('train/images')
    train_labels_dir = Path('train/labels')
    val_images_dir = Path('val/images')
    val_labels_dir = Path('val/labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 各クラスディレクトリを処理
    flowers_path = Path(FLOWERS_DIR)
    total_images = 0
    train_count = 0
    val_count = 0
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = flowers_path / class_name
        
        if not class_dir.exists():
            print(f"警告: {class_dir} が見つかりません。スキップします。")
            continue
        
        # 画像ファイルを取得
        image_files = list(class_dir.glob('*.jpg'))
        total_images += len(image_files)
        
        # train/valに分割
        random.shuffle(image_files)
        split_idx = int(len(image_files) * TRAIN_FRACTION)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # train画像を移動
        for img_path in train_images:
            # 画像をコピー
            dest_img = train_images_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # ラベルファイルを作成（画像全体をバウンディングボックスとして扱う）
            label_path = train_labels_dir / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                # YOLO形式: class_id center_x center_y width height (正規化座標)
                # 画像全体をカバー: center_x=0.5, center_y=0.5, width=1.0, height=1.0
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            train_count += 1
        
        # val画像を移動
        for img_path in val_images:
            # 画像をコピー
            dest_img = val_images_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # ラベルファイルを作成
            label_path = val_labels_dir / (img_path.stem + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            val_count += 1
        
        print(f"{class_name}: train={len(train_images)}, val={len(val_images)}")
    
    print(f"\n完了!")
    print(f"総画像数: {total_images}")
    print(f"train: {train_count} 画像")
    print(f"val: {val_count} 画像")
    print(f"\nディレクトリ構造:")
    print(f"  train/images/ - {len(list(train_images_dir.glob('*.jpg')))} 画像")
    print(f"  train/labels/ - {len(list(train_labels_dir.glob('*.txt')))} ラベル")
    print(f"  val/images/ - {len(list(val_images_dir.glob('*.jpg')))} 画像")
    print(f"  val/labels/ - {len(list(val_labels_dir.glob('*.txt')))} ラベル")

if __name__ == '__main__':
    organize_yolo_dataset()

