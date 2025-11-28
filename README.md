# YOLOv11 による物体検出＋ GrandingDINO による自動アノテーションノートブック

## 必要要件

- Python 3.12
- uv（Python パッケージマネージャー）
- （オプション）gcc（GroundingDINO を mpc 対応にするために C++拡張を使用する）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/ikomiki/YOLO-flower
cd YOLO-flower
```

### 2. 依存関係のインストール

このプロジェクトは`uv`を使用して依存関係を管理しています。

```bash
# 依存関係のインストール
uv sync
```

### 3. GroundingDINO モデルのダウンロード

```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 4. ノートブックの設定を変更する

`YOLO-flower.ipynb` の第 1 番セルにまとめられている設定を適宜変更する。
とくに設定変更が想定されるのは以下に示す項目のみであり、ほかは必要が無ければ変更の必要はない。。

- `MAX_IMAGES_PER_CLASS`: クラスあたりの学習用画像の最大数（None の場合は全ての画像を使用）
- `DEVICE`: cpu, cuda, あるいは mps(Metal Performance Shaders, Apple シリコンの機能)。基本的には自動認識されるが、mps については検証不足が指摘されているので問題が起きた場合は cpu に設定し直す。

### 5. ノートブックを上から実行

`YOLO-flower.ipynb` で「すべてを実行」を実行するか、あるいはセルを上から実行する。
カーネルを指定する場合は、本プロジェクト内の venv を指定する。

処理終了後、`Results saved to`で続くパスに結果が書き込まれる。
