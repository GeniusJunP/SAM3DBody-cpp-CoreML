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

## 5. 制限事項

本リポジトリのCoreMLエクスポートは、バッチサイズ1（`B=1`）に固定されています。
これは、PyTorchの `torch.jit.trace` を用いた際、Vision Transformer特有の空間次元からトークン列への展開処理（例: `[B, C, H, W]` から `[B, H*W, C]` への `reshape` や `flatten`）において、動的であるべきバッチ次元 `B` が計算済みの定数として固定化される仕様と、元のPyTorchモデルの制御フローが `torch.jit.script` に非対応であることに起因します。

この制約により、C++推論エンジン側ではバッチ処理による並列化が行えず、以下のように検出された人数分だけシーケンシャルに推論を実行します。  
  
[src/coreml_backbone.mm](file:///Users/junpei/working/_testCode/SAM3DBody-cpp/src/coreml_backbone.mm#L125-L131)
```cpp
        for (int b = 0; b < batch; ++b) {
            NSError* error = nil;
            std::memcpy(impl_->cached_input.dataPointer,
                        input_nchw + (size_t)b * input_elems,
                        input_elems * sizeof(float));

            id<MLFeatureProvider> result = [impl_->model predictionFromFeatures:impl_->cached_provider error:&error];
```

結果として、処理時間は画面内の人数に比例して増加します。実運用でのパフォーマンス低下を防ぐため、多数の人物が写るシーンでは `--max-persons` 引数による人数上限の設定を推奨します。
