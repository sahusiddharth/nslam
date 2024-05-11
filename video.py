import cv2
import os
import re

def extract_number(filename):
    return int(re.search(r'\d+', filename).group())

# Sort the filenames based on the numeric part



def images_to_video(image_folder, output_video):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    images = sorted(images, key=extract_number)
    print(images)

    # Sort the images by filename

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    image_folder = "cam_output_thread1"
    output_video = "thread.mp4"

    images_to_video(image_folder, output_video)

