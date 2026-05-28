# SAM-3D-Body CoreML Export & Build Guide

このディレクトリは、Apple Silicon (M1/M2/M3) のGPUおよびNeural Engine(ANE)を活用し、推論を行うためのCoreMLネイティブモデルエクスポートスクリプトを提供します。

## 1. 環境構築 (Python)
ONNXモデルをCoreMLネイティブの `.mlpackage` に変換するため、Python環境を構築します。
※ `coremltools 9.0` は `torch==2.7.0` までテストされています。

```bash
uv venv .venv311 --python 3.11
uv pip install -r requirements.txt
```

## 2. CoreMLモデル群の一括変換 (make models)
ルートディレクトリの `Makefile` を使用して全モデル（Backbone, Decoder, YOLO）を一括エクスポート可能です。

```bash
cd ..
make models BACKBONE_SIZE=1024
```

これにより、内部で以下のスクリプトが順次実行されます：
- `export_coreml_backbone.py --size 1024`
- `export_coreml_decoder.py --size 1024`
- `export_coreml_yolo.py`

エクスポート後、Xcodeの `coremlcompiler` によって `.mlmodelc` にコンパイルされ、推論時にロード可能な状態になります。

## 3. C++ ビルド
macOS向けに `CoreML` と `Foundation` フレームワークをリンクし、ビルドを行います。

```bash
make clean
make BACKBONE_SIZE=1024
```

## 4. 推論テスト
ロードしたCoreMLモデルを使用し、姿勢推定パイプラインを通します。

```bash
./build/fast_sam_3dbody_run \
  --from ../person.jpg \
  --coreml-backbone coreml_export/backbone_coreml.mlpackage \
  --coreml-decoder coreml_export/decoder_coreml.mlpackage \
  --coreml-yolo coreml_export/yolo_coreml.mlpackage \
  --headless
```


