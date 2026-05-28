import argparse
import os
import sys
import torch
import coremltools as ct
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11m-pose.pt", help="YOLO model path or name")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # Monkeypatch coremltools to fix TypeError with 1D array int cast
    import coremltools.converters.mil.frontend.torch.ops as coreml_ops
    old_cast = coreml_ops._cast
    def patched_cast(context, node, dtype, builtin_dtype):
        try:
            old_cast(context, node, dtype, builtin_dtype)
        except TypeError:
            from coremltools.converters.mil.mil import Builder as mb
            x = context[node.inputs[0]]
            val = x.val
            if hasattr(val, "item"): val = val.item()
            elif hasattr(val, "flatten"): val = val.flatten()[0]
            res = mb.const(val=dtype(val), name=node.name)
            context.add(res)
    coreml_ops._cast = patched_cast

    print(f"Exporting to TorchScript first (imgsz={args.imgsz}, half=True)...")
    ts_path = model.export(
        format="torchscript",
        imgsz=args.imgsz,
        half=True,
        nms=False,
    )
    print(f"Exported TorchScript to: {ts_path}")

    print("Loading TorchScript and converting to CoreML...")
    traced = torch.jit.load(ts_path)
    
    example_shape = (1, 3, args.imgsz, args.imgsz)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=example_shape)],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        skip_model_load=True,
    )
    
    target_path = os.path.join(args.out_dir, "yolo_coreml.mlpackage")
    print(f"saving {target_path}", flush=True)
    mlmodel.save(target_path)
    print(f"saved {target_path}", flush=True)

if __name__ == "__main__":
    main()
