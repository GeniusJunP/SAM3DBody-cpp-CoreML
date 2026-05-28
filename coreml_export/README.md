# SAM-3D-Body CoreML Export & Build Guide

Apple Silicon (M1/M2/M3) 向けにCoreMLネイティブモデルをエクスポート・ビルドするためのガイドです。

以下のコマンドはすべてリポジトリルート（`SAM3DBody-cpp/`）から実行します。

## 1. 環境構築

※ `coremltools 9.0` は `torch==2.7.0` までテストされています。

```bash
uv venv coreml_export/.venv311 --python 3.11
source coreml_export/.venv311/bin/activate
uv pip install -r coreml_export/requirements.txt
```

変換スクリプトはオリジナルのPythonコード（`AmmarkoV/Fast-SAM-3D-Body`）とPyTorchのチェックポイントに依存しています。

```bash
# オリジナルのPythonリポジトリをクローン
git clone --depth 1 https://github.com/AmmarkoV/Fast-SAM-3D-Body.git

# PyTorchのチェックポイントをダウンロード
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3', local_dir='coreml_export/checkpoints/sam-3d-body-dinov3')"
```

## 2. CoreMLモデルのエクスポート

```bash
make models BACKBONE_SIZE=1024
```

エクスポート後、Xcodeの `coremlcompiler` によって `.mlmodelc` にコンパイルされます。

## 3. C++ ビルド

```bash
make clean
make BACKBONE_SIZE=1024
```

## 4. 推論テスト

```bash
./build/fast_sam_3dbody_run \
  --from ../person.jpg \
  --coreml-backbone coreml_export/backbone_coreml.mlpackage \
  --coreml-decoder coreml_export/decoder_coreml.mlpackage \
  --coreml-yolo coreml_export/yolo_coreml.mlpackage \
  --headless
```
