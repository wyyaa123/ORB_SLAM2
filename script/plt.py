import argparse

import cv2
import matplotlib.pyplot as plt


def show_image(image_path, cmap=None, save_path=None):
	image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	if image is None:
		raise FileNotFoundError("Failed to read image: {}".format(image_path))

	if len(image.shape) == 3:
		if image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		elif image.shape[2] == 4:
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

	plt.figure(figsize=(10, 6))
	if len(image.shape) == 2:
		plt.imshow(image, cmap=cmap if cmap else "gray")
	else:
		plt.imshow(image)
	plt.title(image_path)
	plt.axis("off")

	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches="tight")
	plt.show()


def main():
	parser = argparse.ArgumentParser(description="Display image with matplotlib")
	parser.add_argument("--image", default="datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_semantic/merged/1777018836126185656.png", help="Path to input image")
	parser.add_argument("--cmap", default=None, help="Colormap for grayscale/depth image")
	parser.add_argument("--save", default=None, help="Optional path to save plotted figure")
	args = parser.parse_args()

	show_image(args.image, cmap=args.cmap, save_path=args.save)


if __name__ == "__main__":
	main()
	
