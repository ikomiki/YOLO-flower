# flowers_photos/ ディレクトリの内容を、YOLOの学習用ディレクトリに移動する。
import shutil
from pathlib import Path
import json
import hashlib

# クラス名のマッピング（ディレクトリ名 -> クラスID）
CLASS_MAPPING = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}

# Grounding-DINOモデルのロード（初回のみ）
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"
grounding_dino_model = None

def load_grounding_dino():
    """Grounding-DINOモデルをロード"""
    global grounding_dino_model
    
    if not GROUNDING_DINO_AVAILABLE:
        return None
    
    if grounding_dino_model is None:
        try:
            # 設定ファイルとチェックポイントのパスを確認
            config_path = GROUNDING_DINO_CONFIG_PATH
            checkpoint_path = GROUNDING_DINO_CHECKPOINT_PATH
            
            # パスが存在しない場合は、デフォルトパスを試す
            if not os.path.exists(config_path):
                config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            if not os.path.exists(checkpoint_path):
                checkpoint_path = "groundingdino_swint_ogc.pth"
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
            
            grounding_dino_model = load_model(config_path, checkpoint_path)
            print("Grounding-DINOモデルをロードしました")
        except Exception as e:
            print(f"Grounding-DINOモデルのロードに失敗しました: {e}")
            print("groundingdino_swint_ogc.pthをダウンロードしてください")
            print("  wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
            return None
    return grounding_dino_model

def get_cache_path(image_path, class_name):
    """キャッシュファイルのパスを取得"""
    cache_dir = Path('.cache/annotations')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像パスとクラス名からハッシュを生成
    cache_key = f"{image_path}_{class_name}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return cache_dir / f"{cache_hash}.json"

def get_annotation_with_grounding_dino(image_path, class_name, class_id, debug=False, min_confidence=0.3):
    """Grounding-DINOを使用してバウンディングボックスを取得（キャッシュ対応）
    
    Args:
        image_path: 画像のパス
        class_name: クラス名（プロンプトとして使用）
        class_id: クラスID
        debug: デバッグ情報を表示するか
        min_confidence: 最小信頼度（この値以上のバウンディングボックスを使用）
    
    Returns:
        (success, bboxes) - successがTrueの場合、bboxesはYOLO形式の正規化座標のリスト [[center_x, center_y, width, height], ...]
                            失敗した場合はNone
    """
    if not GROUNDING_DINO_AVAILABLE:
        return False, None
    
    # キャッシュファイルのパスを取得
    cache_path = get_cache_path(image_path, class_name)
    
    # キャッシュが存在する場合は読み込む
    cached_boxes = None
    cached_logits = None
    cached_phrases = None
    
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                if debug:
                    print(f"  Debug: キャッシュから読み込み: {image_path.name}")
                # キャッシュからpredict結果を復元
                if 'boxes' in cache_data and 'logits' in cache_data and 'phrases' in cache_data:
                    cached_boxes = torch.tensor(cache_data['boxes'])
                    cached_logits = torch.tensor(cache_data['logits'])
                    cached_phrases = cache_data['phrases']
        except Exception as e:
            if debug:
                print(f"  Debug: キャッシュ読み込みエラー: {e}")
            # キャッシュが破損している場合は再計算
    
    # キャッシュがない場合は予測を実行
    if cached_boxes is None:
        model = load_grounding_dino()
        if model is None:
            return False, None
        
        try:
            # 画像を読み込み
            image_source, image = load_image(str(image_path))
            
            # テキストプロンプト（複数の形式を試す）
            # Grounding-DINOは "object ." の形式を好む
            text_prompt = f"{class_name} ."
            box_threshold = 0.15  # より低いしきい値に変更
            text_threshold = 0.10  # より低いしきい値に変更
            
            # 予測（DEVICE定数を使用）
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=DEVICE
            )
            
            # predict直後にキャッシュに保存
            try:
                cache_data = {
                    'boxes': boxes.cpu().tolist() if isinstance(boxes, torch.Tensor) else boxes.tolist() if hasattr(boxes, 'tolist') else boxes,
                    'logits': logits.cpu().tolist() if isinstance(logits, torch.Tensor) else logits.tolist() if hasattr(logits, 'tolist') else logits,
                    'phrases': phrases
                }
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f)
                if debug:
                    print(f"  Debug: キャッシュに保存: {image_path.name}")
            except Exception as e:
                if debug:
                    print(f"  Debug: キャッシュ保存エラー: {e}")
            
            # キャッシュ変数に設定
            cached_boxes = boxes
            cached_logits = logits
            cached_phrases = phrases
            
        except Exception as e:
            if debug:
                print(f"  Error: {image_path.name} - {e}")
                import traceback
                traceback.print_exc()
            return False, None
    
    # キャッシュから読み込んだデータまたは予測結果を使用
    boxes = cached_boxes
    logits = cached_logits
    phrases = cached_phrases
    
    if debug:
        print(f"  Debug: {image_path.name} - boxes: {len(boxes)}, phrases: {phrases[:3] if len(phrases) > 0 else 'none'}")
    
    try:
        
        # バウンディングボックスが取得できた場合
        if len(boxes) > 0:
            # boxesとlogitsの形状を確認
            # boxesは (n, 4) の形状、logitsは (n,) の形状のはず
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu()
            if isinstance(logits, torch.Tensor):
                logits = logits.cpu()
            
            # boxesが2次元テンソルであることを確認
            if boxes.dim() == 1:
                # 1次元の場合は (4,) なので、1つのボックスとして扱う
                boxes = boxes.unsqueeze(0)
            elif boxes.dim() == 0:
                # 0次元の場合はエラー
                if debug:
                    print(f"  Debug: boxesが0次元です。形状: {boxes.shape}")
                return False, None
            
            # logitsが1次元であることを確認
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            elif logits.dim() > 1:
                # 2次元以上の場合はmaxを取る
                logits = logits.max(dim=1)[0]
            
            # 信頼度が閾値以上のバウンディングボックスを全て取得
            valid_boxes = []
            # boxesとlogitsの長さを確認
            num_boxes = boxes.shape[0] if boxes.dim() > 0 else 1
            num_logits = logits.shape[0] if logits.dim() > 0 else 1
            min_len = min(num_boxes, num_logits)
            
            for idx in range(min_len):
                # boxを取得（形状が (4,) であることを確認）
                if boxes.dim() == 2:
                    box = boxes[idx]  # (4,)
                elif boxes.dim() == 1:
                    box = boxes  # 既に (4,)
                else:
                    if debug:
                        print(f"  Debug: 予期しないboxesの形状: {boxes.shape}")
                    continue
                
                # logitを取得
                if logits.dim() == 1:
                    logit = logits[idx]
                elif logits.dim() == 0:
                    logit = logits
                else:
                    if debug:
                        print(f"  Debug: 予期しないlogitsの形状: {logits.shape}")
                    continue
                
                # logitの値を取得
                if isinstance(logit, torch.Tensor):
                    logit_value = logit.item()
                else:
                    logit_value = float(logit)
                
                if logit_value >= min_confidence:
                    # boxをnumpy配列に変換
                    if isinstance(box, torch.Tensor):
                        box_np = box.numpy()
                    else:
                        box_np = np.array(box)
                    
                    # box_npが1次元配列（4要素）であることを確認
                    if box_np.ndim == 0:
                        if debug:
                            print(f"  Debug: box_npが0次元です。box={box}, box_np={box_np}")
                        continue
                    elif box_np.ndim > 1:
                        box_np = box_np.flatten()
                    
                    if len(box_np) != 4:
                        if debug:
                            print(f"  Debug: box_npの要素数が4ではありません: {len(box_np)}")
                        continue
                    
                    center_x, center_y, width, height = box_np
                    
                    # 座標を0-1の範囲にクランプ
                    center_x = max(0.0, min(1.0, float(center_x)))
                    center_y = max(0.0, min(1.0, float(center_y)))
                    width = max(0.001, min(1.0, float(width)))  # 最小値を0.001に設定
                    height = max(0.001, min(1.0, float(height)))  # 最小値を0.001に設定
                    
                    # バウンディングボックスが画像内に収まっていることを確認
                    x_min = center_x - width / 2.0
                    x_max = center_x + width / 2.0
                    y_min = center_y - height / 2.0
                    y_max = center_y + height / 2.0
                    
                    if 0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1:
                        valid_boxes.append([center_x, center_y, width, height])
                        if debug:
                            print(f"  Debug: valid box[{idx}]={[center_x, center_y, width, height]}, logit={logit_value:.3f}")
            
            if len(valid_boxes) > 0:
                if debug:
                    print(f"  Debug: {len(valid_boxes)}個の有効なバウンディングボックスを取得")
                return True, valid_boxes
            elif debug:
                print(f"  Debug: 信頼度{min_confidence}以上のバウンディングボックスが見つかりませんでした")
        
        # バウンディングボックスが取得できなかった場合
        if debug:
            print(f"  Debug: バウンディングボックスが見つかりませんでした")
        return False, None
        
    except Exception as e:
        # デバッグモードの場合はエラーを表示
        if debug:
            print(f"  Error: {image_path.name} - {e}")
            import traceback
            traceback.print_exc()
        return False, None

def organize_yolo_dataset():
    """flower_photosディレクトリの内容をYOLOの学習形式に変換
    
    - 各ディレクトリ（daisy, dandelion, roses, sunflowers, tulips）ごとにクラス番号を割り振る
    - train/valに分割して移動
    - 各画像に対応するラベルテキストファイルを作成（アノテーション範囲は画面の全領域）
    """
    
    # ランダムシードを設定
    random.seed(RANDOM_SEED)
    
    # 出力ディレクトリを作成（dataset/以下に配置）
    train_images_dir = Path('dataset/train/images')
    train_labels_dir = Path('dataset/train/labels')
    val_images_dir = Path('dataset/val/images')
    val_labels_dir = Path('dataset/val/labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Grounding-DINOモデルをロード
    print("Grounding-DINOモデルをロード中...")
    load_grounding_dino()
    
    # 各クラスディレクトリを処理
    flowers_path = Path(FLOWERS_DIR)
    total_images = 0
    train_count = 0
    val_count = 0
    annotation_success_count = 0
    annotation_fail_count = 0
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = flowers_path / class_name
        
        if not class_dir.exists():
            print(f"警告: {class_dir} が見つかりません。スキップします。")
            continue
        
        # 画像ファイルを取得
        image_files = list(class_dir.glob('*.jpg'))
        # # 動作確認のため、クラスあたり5枚だけ処理
        # MAX_IMAGES_PER_CLASS = 5
        # if len(image_files) > MAX_IMAGES_PER_CLASS:
        #     image_files = image_files[:MAX_IMAGES_PER_CLASS]
        #     print(f"  注意: 動作確認のため、{class_name}は{MAX_IMAGES_PER_CLASS}枚のみ処理します")
        total_images += len(image_files)
        
        # ランダムにシャッフル
        random.shuffle(image_files)
        
        # 成功した画像と失敗した画像を分けて処理
        success_images = []  # (img_path, bbox) のリスト
        fail_images = []  # img_path のリスト
        
        # 各画像を処理してアノテーションを取得（進捗バー付き）
        print(f"\n{class_name} (クラスID: {class_id}): アノテーション取得中...")
        for idx, img_path in enumerate(tqdm(image_files, desc=f"  {class_name}", unit="画像")):
            # 最初の数枚の画像でデバッグ情報を表示
            debug_mode = (idx < 3) and (class_name == list(CLASS_MAPPING.keys())[0])
            
            # Grounding-DINOでアノテーションを取得
            success, bboxes = get_annotation_with_grounding_dino(img_path, class_name, class_id, debug=debug_mode)
            
            if success:
                success_images.append((img_path, bboxes))
                annotation_success_count += 1
            else:
                fail_images.append(img_path)
                annotation_fail_count += 1
        
        # 成功した画像をtrain/valに分割
        random.shuffle(success_images)
        split_idx = int(len(success_images) * TRAIN_FRACTION)
        success_train = success_images[:split_idx]
        success_val = success_images[split_idx:]
        
        # 成功した画像のtrain分を処理（進捗バー付き）
        if len(success_train) > 0:
            print(f"  train用画像をコピー中... ({len(success_train)} 画像)")
            # YOLOは1画像あたり最大300個のバウンディングボックスを処理可能
            MAX_BOXES_PER_IMAGE = 300
            for img_path, bboxes in tqdm(success_train, desc="    train", unit="画像", leave=False):
                dest_img = train_images_dir / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # ラベルファイルを作成（複数のバウンディングボックスに対応）
                label_path = train_labels_dir / (img_path.stem + '.txt')
                # Python側で文字列を結合してから一括書き込み（性能向上のため）
                label_lines = [
                    f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    for bbox in bboxes[:MAX_BOXES_PER_IMAGE]
                ]
                # Python側で文字列を結合してから一括書き込み（性能向上のため）
                label_lines = [
                    f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    for bbox in bboxes[:MAX_BOXES_PER_IMAGE]
                ]
                with open(label_path, 'w') as f:
                    f.writelines(label_lines)
                
                val_count += 1
        
        # 失敗した画像は全てvalに追加（画像全体を範囲として）（進捗バー付き）
        if len(fail_images) > 0:
            print(f"  失敗画像をvalにコピー中... ({len(fail_images)} 画像)")
            for img_path in tqdm(fail_images, desc="    val(fail)", unit="画像", leave=False):
                dest_img = val_images_dir / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # ラベルファイルを作成（アノテーション範囲は画面の全領域）
                label_path = val_labels_dir / (img_path.stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                val_count += 1
        
        print(f"  {class_name} (クラスID: {class_id}): 処理完了 ({len(image_files)} 画像)")
    
    print(f"\n完了!")
    print(f"総画像数: {total_images}")
    print(f"train: {train_count} 画像")
    print(f"val: {val_count} 画像")
    print(f"\nアノテーション統計:")
    print(f"  成功: {annotation_success_count} 画像 (うち {train_count} 画像をtrain、{annotation_success_count - train_count} 画像をvalに配置)")
    print(f"  失敗: {annotation_fail_count} 画像 (全てvalに配置、画像全体を範囲として使用)")
    print(f"\nディレクトリ構造:")
    print(f"  dataset/train/images/ - {len(list(train_images_dir.glob('*.jpg')))} 画像")
    print(f"  dataset/train/labels/ - {len(list(train_labels_dir.glob('*.txt')))} ラベル")
    print(f"  dataset/val/images/ - {len(list(val_images_dir.glob('*.jpg')))} 画像")
    print(f"  dataset/val/labels/ - {len(list(val_labels_dir.glob('*.txt')))} ラベル")

# 実行
organize_yolo_dataset()

