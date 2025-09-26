# Multi-Person Tracking Model

マルチチャンネルヒートマップデータを用いた複数人物位置推定モデル

## 動作環境

### Python バージョン
- Python 3.7以上（推奨: Python 3.8 - 3.10）

### 必要なライブラリ

以下のライブラリが必要です：

```bash
# 基本的な依存関係
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0

# PyTorchのインストール（CUDA対応版を推奨）
# CPU版の場合:
pip install torch torchvision

# CUDA 11.8版の場合（GPUを使用する場合）:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### インストール手順

1. 仮想環境の作成（推奨）:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

2. 必要なライブラリのインストール:
```bash
pip install torch numpy matplotlib
```

## 使用方法

### データ構造
データは以下の構造で配置してください：
```
./dataset3/
├── train/     # トレーニング用JSONファイル
├── val/       # 検証用JSONファイル
└── test/      # テスト用JSONファイル
```

### 実行方法

1. **フルパイプライン実行**（トレーニング + 推論）:
```bash
python main.py --mode both
```

2. **トレーニングのみ**:
```bash
python main.py --mode train --epochs 100 --batch_size 8
```

3. **推論のみ**:
```bash
python main.py --mode inference --model_path ./outputs/best_model.pth
```

4. **デバッグモード**:
```bash
python main.py --mode debug
```

### コマンドライン引数

- `--mode`: 実行モード（train, inference, both, debug）
- `--data_dir`: データセットのベースディレクトリ（デフォルト: ./dataset3）
- `--epochs`: トレーニングエポック数（デフォルト: 100）
- `--batch_size`: バッチサイズ（デフォルト: 8）
- `--lr`: 学習率（デフォルト: 0.001）
- `--model_path`: モデルの保存/読み込みパス（デフォルト: ./outputs/best_model.pth）
- `--test_json`: テスト用JSONファイルパス

### 出力ファイル

実行後、以下のファイルが生成されます：

- `./outputs/best_model.pth`: 最良モデル
- `./outputs/training_curve.png`: トレーニング曲線
- `./outputs/inference_visualization.png`: 推論結果の可視化
- `./outputs/inference_results.csv`: 推論結果のCSVファイル

## トラブルシューティング

### CUDA関連のエラー
GPUが利用できない場合、自動的にCPUモードで実行されます。

### メモリ不足エラー
バッチサイズを小さくしてください：
```bash
python main.py --batch_size 4
```

### データセットエラー
JSONファイルの形式が正しいか確認してください。各JSONファイルには以下の構造が必要です：
- `windows`: 時系列ウィンドウのリスト
- `sensors`: センサーデータ
- `ground_truth`: 正解位置データ
