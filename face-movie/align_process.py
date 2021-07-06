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

def get_landmarks(im, target_rect):
    res = np.matrix([[p.x, p.y] for p in PREDICTOR(im, target_rect).parts()])
    return res
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname, rects):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    s = get_landmarks(im, rects)
    return im, s

def warp_im(im, M, dshape, prev):
    output_im = cv2.warpAffine(
        im, M, (dshape[1], dshape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101 if prev is not None else cv2.BORDER_CONSTANT,
    )

    if prev is not None:
        # overlay the image on the previous images
        mask = cv2.warpAffine(
            np.ones_like(im, dtype='float32'), M, 
            (dshape[1], dshape[0]), flags=cv2.INTER_CUBIC,
        )
        output_im = mask * output_im + (1-mask) * prev

    return output_im


def align_images(impath1, impath2, border, prev=None):
    filename = os.path.basename(impath2).split('.')[0]
    ext = os.path.basename(impath2).split('.')[1]
    read_rects = "{}/rects/{}.{}.rects".format(OUTPUT_DIR, filename, ext)
    with open(read_rects, 'rb') as fp:
        rects = pickle.load(fp)
    if len(rects)==0:
        return prev
    elif len(rects)>1:
        rects=rects[answers["{}_annotated.jpg".format(im)]]
    elif len(rects)==1:
        rects=rects[0]
    im2, landmarks2 = read_im_and_landmarks(impath2, rects)

    filename = os.path.basename(impath2).split('.')[0]
    ext = os.path.basename(impath2).split('.')[1]
    
    T = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    M = cv2.invertAffineTransform(T[:2])

    if border is not None:
        im2 = cv2.copyMakeBorder(im2, border, border, border, border, 
            borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

    warped_im2 = warp_im(im2, M, im1.shape, prev)
    my_text = "{}/{}/{}".format(filename[6:8],filename[4:6],filename[0:4])
    my_font_face = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    my_font_scale = 2
    my_font_thickness = 4
    my_text_size = cv2.getTextSize(my_text, my_font_face, my_font_scale, my_font_thickness)

    warped_im2_with_date = warped_im2.copy()
    cv2.putText(warped_im2_with_date, 
                    my_text, 
                    (0,warped_im2_with_date.shape[0] - my_text_size[1]),
                    #(0,500),
                    fontFace=my_font_face,
                    fontScale=my_font_scale,
                    color=(0, 0, 0),
                    thickness=my_font_thickness+15)
    cv2.putText(warped_im2_with_date, 
                    my_text, 
                    (0,warped_im2_with_date.shape[0] - my_text_size[1]),
                    #(0,500),
                    fontFace=my_font_face,
                    fontScale=my_font_scale,
                    color=(255, 255, 255),
                    thickness=my_font_thickness)

    cv2.imwrite("{}/{}.{}.jpg".format(OUTPUT_DIR, filename,ext), warped_im2_with_date)
    print("Aligned {}".format(filename))
    return warped_im2

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

    answers_file = "{}/rects/answers".format(OUTPUT_DIR)
    with open (answers_file, 'rb') as fp:
        answers = pickle.load(fp)

    valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic"]
    get_ext = lambda f: os.path.splitext(f)[1].lower()

    prev = None

    # Constraints on input images (for aligning):
    # - Must have clear frontal view of a face (there may be multiple)
    # - Filenames must be in lexicographic order of the order in which they are to appear  
    
    im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
    im_files = sorted(im_files, key=lambda x: x.split('/'))

    filename = os.path.basename(target).split('.')[0]
    ext = os.path.basename(target).split('.')[1]
    read_rects = "{}/rects/{}.{}.rects".format(OUTPUT_DIR, filename, ext)
    with open(read_rects, 'rb') as fp:
        rects = pickle.load(fp)
    im1, landmarks1 = read_im_and_landmarks(target, rects[0])

    for im in im_files:
        if overlay:
            prev = align_images(target, im_dir + '/' + im, border, prev)
        else:
            align_images(target, im_dir + '/' + im, border)