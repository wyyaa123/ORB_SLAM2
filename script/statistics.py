from pathlib import Path

# 你的数据集根目录（包含 rgb 文件夹）
dataset_dir = Path("datasets/Real/D435I/ot")
img_dir = dataset_dir / "depth"  # 这里放 png
out_path = dataset_dir / "depth.txt"

lines = [
    "# depth images",
    "# file: 'your.bag'",
    "# timestamp filename",
]

for p in sorted(img_dir.glob("*.png")):
    ts = p.stem  # 用文件名当时间戳
    rel = p.relative_to(dataset_dir).as_posix()  # 生成 depth/xxx.png
    lines.append(f"{ts} {rel}")

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")