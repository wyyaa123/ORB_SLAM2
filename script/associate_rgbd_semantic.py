#!/usr/bin/env python3
"""
Associate RGB, depth, and semantic images by timestamp and write them to one file.
"""

import argparse
import os
import sys


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _has_images(path):
    try:
        for name in os.listdir(path):
            _, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS and os.path.isfile(os.path.join(path, name)):
                return True
    except FileNotFoundError:
        return False
    return False


def _gather_images(path, recursive=False):
    images = []
    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                _, ext = os.path.splitext(name)
                if ext.lower() in IMAGE_EXTS:
                    images.append(os.path.join(root, name))
        return images

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        _, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS and os.path.isfile(full_path):
            images.append(full_path)
    return images


def _parse_stamp_from_filename(path):
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    try:
        return float(stem)
    except ValueError:
        return None


def _build_stamp_map(paths, base_dir):
    stamp_map = {}
    for path in paths:
        stamp = _parse_stamp_from_filename(path)
        if stamp is None:
            continue
        rel_path = os.path.relpath(path, base_dir)
        if stamp in stamp_map:
            continue
        stamp_map[stamp] = rel_path
    return stamp_map


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two sorted timestamp lists using a greedy two-pointer scan.
    Returns a list of (stamp1, stamp2).
    """
    first_keys = sorted(first_list)
    second_keys = sorted(second_list)

    i = 0
    j = 0
    matches = []
    while i < len(first_keys) and j < len(second_keys):
        t1 = first_keys[i]
        t2 = second_keys[j] + offset
        diff = t1 - t2
        if abs(diff) <= max_difference:
            matches.append((first_keys[i], second_keys[j]))
            i += 1
            j += 1
        elif diff > 0:
            j += 1
        else:
            i += 1

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Associate RGB, depth, and semantic images by timestamp."
    )
    parser.add_argument("rgb_dir", default="datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_image_rect_raw", help="RGB image folder")
    parser.add_argument("depth_dir", default="datasets/Real/D435I/2026-04-24-08-20-35/depth", help="Depth image folder")
    parser.add_argument("semantic_dir", default="datasets/Real/D435I/2026-04-24-08-20-35/semantic", help="Semantic image folder (or parent)")
    parser.add_argument(
        "--depth_offset",
        default=0.0,
        type=float,
        help="Time offset applied to depth timestamps (seconds)",
    )
    parser.add_argument(
        "--semantic_offset",
        default=0.0,
        type=float,
        help="Time offset applied to semantic timestamps (seconds)",
    )
    parser.add_argument(
        "--max_difference",
        default=0.02,
        type=float,
        help="Max allowed time difference (seconds)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--recursive_semantic",
        action="store_true",
        help="Search semantic_dir recursively for images",
    )
    parser.add_argument(
        "--base_dir",
        default="",
        help="Base directory for relative paths (default: common parent)",
    )
    args = parser.parse_args()

    rgb_dir = args.rgb_dir
    depth_dir = args.depth_dir
    semantic_dir = args.semantic_dir

    if not _has_images(semantic_dir):
        merged_dir = os.path.join(semantic_dir, "merged")
        if _has_images(merged_dir):
            semantic_dir = merged_dir

    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = os.path.commonpath([rgb_dir, depth_dir, semantic_dir])

    rgb_paths = _gather_images(rgb_dir)
    depth_paths = _gather_images(depth_dir)
    semantic_paths = _gather_images(semantic_dir, recursive=args.recursive_semantic)

    rgb_map = _build_stamp_map(rgb_paths, base_dir)
    depth_map = _build_stamp_map(depth_paths, base_dir)
    semantic_map = _build_stamp_map(semantic_paths, base_dir)

    rgb_depth = associate(
        rgb_map.keys(), depth_map.keys(), args.depth_offset, args.max_difference
    )
    rgb_semantic = associate(
        rgb_map.keys(), semantic_map.keys(), args.semantic_offset, args.max_difference
    )

    depth_by_rgb = {t_rgb: t_depth for t_rgb, t_depth in rgb_depth}
    semantic_by_rgb = {t_rgb: t_sem for t_rgb, t_sem in rgb_semantic}

    if args.output == "-":
        out_f = sys.stdout
        close_out = False
    else:
        out_f = open(args.output, "w", encoding="utf-8")
        close_out = True

    try:
        for t_rgb in sorted(rgb_map.keys()):
            if t_rgb not in depth_by_rgb or t_rgb not in semantic_by_rgb:
                continue
            t_depth = depth_by_rgb[t_rgb]
            t_sem = semantic_by_rgb[t_rgb]
            out_f.write(
                "{:.6f} {} {:.6f} {} {:.6f} {}\n".format(
                    t_rgb,
                    rgb_map[t_rgb],
                    t_depth - args.depth_offset,
                    depth_map[t_depth],
                    t_sem - args.semantic_offset,
                    semantic_map[t_sem],
                )
            )
    finally:
        if close_out:
            out_f.close()


if __name__ == "__main__":
    main()
