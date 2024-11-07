import cv2

from tinyface.typing import VisionFrame


def read_image(image_path: str) -> VisionFrame:
    return cv2.imread(image_path)


def write_image(image_path: str, vision_frame: VisionFrame) -> bool:
    if image_path:
        return cv2.imwrite(image_path, vision_frame)
    return False


def show_image(vision_frame: VisionFrame) -> None:
    cv2.imshow("Preview", vision_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
