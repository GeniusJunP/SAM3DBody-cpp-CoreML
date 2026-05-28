# CoreML Backbone Export & Build Guide

## 1. 環境構築 (Python)
バックボーンのONNXモデルをCoreMLネイティブの `.mlpackage` に変換するため、Python環境を構築します。
```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt # (torch, coremltools等)
```

## 2. CoreMLモデルへの変換
YOLOで切り出された画像 ([1, 3, 512, 512]) を入力とするため、動的形状を固定し `aten::Int` の変換エラーを回避するパッチを当てたエクスポートスクリプトを実行します。
```bash
uv run --python .venv/bin/python export_coreml_backbone.py
```
成功すると `backbone_coreml.mlpackage` が生成されます。

## 3. macOS ネイティブ事前コンパイル (任意)
毎回の推論開始時のロード時間（約5-6秒）を短縮するため、Xcodeの `coremlcompiler` で事前コンパイルしておきます。
```bash
xcrun coremlcompiler compile backbone_coreml.mlpackage backbone_coreml_compiled
```

## 4. C++ ビルドと実行
macOS向けに `CoreML` と `Foundation` フレームワークをリンクし、ビルドを行います。
```bash
cmake -S . -B build-coreml \
      -DONNX_RUNTIME_DIR=/opt/homebrew/opt/onnxruntime \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DGGML_METAL=OFF
cmake --build build-coreml --target fast_sam_3dbody_run -j $(sysctl -n hw.ncpu)
```

## 5. 推論テスト
ロードしたCoreMLモデルを使用し、姿勢推定パイプラインを通します。
```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib:build-coreml \
./build-coreml/fast_sam_3dbody_run \
  --from ../doc/screen.jpg \
  --onnx-dir ./onnx_fp32 \
  --coreml-backbone ./backbone_coreml_compiled/backbone_coreml.mlmodelc \
  --cuda -1 --no-fp16 --headless \
  --out ./coreml_pose_lbs_test.csv
```
