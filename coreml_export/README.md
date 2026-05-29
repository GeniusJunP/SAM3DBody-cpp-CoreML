# SAM-3D-Body CoreML Export/Build Guide

Apple Silicon 向けにCoreMLネイティブモデルをエクスポート・ビルドするためのガイドです。

以下のコマンドはすべてリポジトリルート（`SAM3DBody-cpp/`）から実行します。

## 1. 環境構築

※ モデルのコンパイル（`xcrun`）に必要となるため、あらかじめ Xcode Command Line Tools をインストールしておいてください。 `xcode-select --install`  
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
make models BACKBONE_SIZE=512
```

エクスポート後、Xcodeの `coremlcompiler` によって `.mlmodelc` にコンパイルされます。

## 3. C++ ビルド

```bash
make clean
make BACKBONE_SIZE=512
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

> **Note:** 上記の `--coreml-*` 引数群は、他の実行コマンドにもそのまま引き継いで使用できます。
> 例えば、Pythonフロントエンド（UI表示）や、オフラインのBVH書き出しスクリプトを実行する際にも、元の `--onnx-dir` 等の代わりにこれらの引数を付与するだけで、すべてCoreMLによる推論が有効になります。
>
> ```bash
> # Python軽量フロントエンド（ウェブカメラ）での実行例
> python fast_sam_3dbody_frontend.py \
>   --from 0 \
>   --coreml-backbone coreml_export/backbone_coreml.mlpackage \
>   --coreml-decoder coreml_export/decoder_coreml.mlpackage \
>   --coreml-yolo coreml_export/yolo_coreml.mlpackage
>
> # オフラインのマルチパスBVH生成での実行例
> ./build/offline_sam_3dbody_render \
>   --from video.mp4 \
>   --bvh output.bvh \
>   --coreml-backbone coreml_export/backbone_coreml.mlpackage \
>   --coreml-decoder coreml_export/decoder_coreml.mlpackage \
>   --coreml-yolo coreml_export/yolo_coreml.mlpackage
> ```
