"""doclayout_yoloによる事前推論器."""

import os
import glob
from doclayout_yolo import YOLOv10
from PIL import Image


input_jpg_dir = "./input_jpg_files"
output_json_dir = "./output_json_files"

model = YOLOv10(
    "./models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
)


def process(jpg_file_path: str):
    base_name = os.path.splitext(os.path.basename(jpg_file_path))[
        0
    ]  # ファイル名（拡張子なし）

    image = Image.open(jpg_file_path)
    det_res = model.predict(
        image,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cpu",  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    output_file = os.path.join(output_json_dir, f"{base_name}.json")
    with open(output_file, "w") as f:
        f.write(det_res[0].tojson())

    print(f"Created: {output_file}")


def main():
    for jpg_file_path in glob.glob(f"{input_jpg_dir}/*.jpg"):
        print(f"Processing file: {jpg_file_path}")
        process(jpg_file_path)


if __name__ == "__main__":
    os.makedirs(output_json_dir, exist_ok=True)
    main()
