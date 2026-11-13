"""Export TransNetV2's published weights to OpenVINO for ``shot_model``.

    python tools/export_transnetv2.py PATH/TO/transnetv2-weights [OUT_DIR]

``transnetv2-weights`` is the saved-model folder from
github.com/soCzech/TransNetV2 (``inference/transnetv2-weights``: MIT, stored
with git-lfs, about 36 MB). OpenVINO's TensorFlow frontend reads it directly;
TensorFlow itself is not needed. Writes ``transnetv2.xml`` + ``.bin`` (fp16,
about 16 MB) to OUT_DIR, by default ``models/transnetv2`` here.
"""
import os
import sys


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    import openvino as ov
    source = argv[0]
    out_dir = argv[1] if len(argv) > 1 else os.path.join("models", "transnetv2")
    os.makedirs(out_dir, exist_ok=True)
    model = ov.convert_model(source)
    path = os.path.join(out_dir, "transnetv2.xml")
    ov.save_model(model, path, compress_to_fp16=True)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
