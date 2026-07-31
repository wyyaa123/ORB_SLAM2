#!/usr/bin/env python3

"""Remove the leading sequence number from each pose-data row."""

import argparse
from pathlib import Path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(
        f"{input_path.stem}.no_index{input_path.suffix}"
    )


def remove_pose_indices(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出文件不能是同一个文件")

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                destination.write(line)
                continue

            fields = stripped.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"第 {line_number} 行没有可删除的序号")

            try:
                int(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"第 {line_number} 行的第一列不是整数序号: {fields[0]}"
                ) from error

            destination.write(fields[1] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="删除位姿数据每个数据行开头的整数序号。"
    )
    parser.add_argument("input", type=Path, help="原始位姿文件")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="输出文件；省略时在原文件名中加入 .no_index",
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.input)
    remove_pose_indices(args.input, output_path)
    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()
