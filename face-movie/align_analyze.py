# USAGE: python face-movie/align.py -images IMAGES -target TARGET [-overlay] [-border BORDER] -outdir OUTDIR

import cv2
import dlib
import numpy as np
import argparse
import os
import pickle

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)


def prompt_user_to_choose_face(im, rects):
    im = im.copy()
    h, w = im.shape[:2]
    for i in range(len(rects)):
        d = rects[i]
        x1, y1, x2, y2 = d.left(), d.top(), d.right()+1, d.bottom()+1
        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)
        cv2.putText(im, str(i), (d.center().x, d.center().y),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255),
                    thickness=5)

    DISPLAY_HEIGHT = 650
    resized = cv2.resize(im, (int(w * DISPLAY_HEIGHT / float(h)), DISPLAY_HEIGHT))
    write_to = "{}/annotated/{}_annotated.jpg".format(OUTPUT_DIR, im_name)
    if os.path.exists(write_to):
        print("Error: {} exists".format(write_to))
    cv2.imwrite(write_to, resized)
    print("Annotated {}".format(im_name))


def get_landmarks(im):
    rects = DETECTOR(im, 1)
    if len(rects) == 0 and len(DETECTOR(im, 0)) > 0:
        rects = DETECTOR(im, 0)
    if len(rects) > 1:
        prompt_user_to_choose_face(im, rects)
    return rects

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    s = get_landmarks(im)
    return im, s

def align_images(impath1, impath2, border, prev=None):

    im2, landmarks2 = read_im_and_landmarks(impath2)

    filename = os.path.basename(impath2).split('.')[0]
    ext = os.path.basename(impath2).split('.')[1]
    write_to = "{}/rects/{}.{}.rects".format(OUTPUT_DIR, filename, ext)
    if os.path.exists(write_to):
        print("Error: {} exists".format(write_to))
    else:
        with open(write_to, 'wb') as output:
            pickle.dump(landmarks2, output)
        print("Wrote {} - {} face(s)".format(write_to, len(landmarks2)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-images", help="Directory of images to be aligned", required=True)
    ap.add_argument("-target", help="Path to target image to which all others will be aligned", required=True)
    ap.add_argument("-overlay", help="Flag to overlay images on top of each other", action='store_true')
    ap.add_argument("-border", type=int, help="Border size (in pixels) to be added to images")
    ap.add_argument("-outdir", help="Output directory name", required=True)
    args = vars(ap.parse_args())
    im_dir = args["images"]
    target = args["target"]
    overlay = args["overlay"]
    border = args["border"]
    OUTPUT_DIR = args["outdir"]

    valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic"]
    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Constraints on input images (for aligning):
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear  
    align_images(target, target, border)

    im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
    im_files = sorted(im_files, key=lambda x: x.split('/'))
    for im_name in im_files:
        align_images(target, im_dir + '/' + im_name, border)
