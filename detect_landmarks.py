import face_alignment
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import argparse


def run(img_path, txt_path, vis):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd', flip_input=False)
    image = io.imread(img_path)
    det = fa.get_landmarks_from_image(image)

    np.savetxt(txt_path, det[0] ,fmt='%.4f')

    if vis:
        plt.imshow(image)
        for detection in det:
            plt.scatter(detection[:,0], detection[:,1], 2)
    print('Landmarks saved in ' + txt_path)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='2D Landmarks Detection')
    parser.add_argument('--image_path', type=str, help='Path to image')
    parser.add_argument('--txt_path', type=str, help='Path to saved txt')
    parser.add_argument('--vis', type=bool, default=False, help='Visualize landmarks on image')
    args = parser.parse_args()

    run(args.image_path, args.txt_path, args.vis)