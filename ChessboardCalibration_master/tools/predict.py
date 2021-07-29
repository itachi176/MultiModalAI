import pickle 
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from calibration.Functions import distance, roundToPoint5
from calibration import Predictor
import sys
import argparse
def parse_args():
    parser =argparse.ArgumentParser(description="Run calibration")
    parser.add_argument('--image', type=str, help='path to image', default= './calibration/chessboard.jpg')
    parser.add_argument('--calib_data', type=str, help='calibration data', default='./calibration/calib.pkl')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--out_dir', type=str, help='output images path', default='./out_dir/images/')
    args = parser.parse_args()
    return args

def pred(x, y, img):

    calib_data = "./ChessboardCalibration_master/calibration/calib.pkl"
    origin = (625,34)
    out_dir = './ChessboardCalibration_master/out_dir/images/'
    predictor = Predictor(img_path=img, calib_data = calib_data, origin=origin, out_dir=out_dir)

        #print(a,b)
    real_x, real_y = predictor.predict(x, y)
    return real_x, real_y

# if __name__ == "__main__":
#     main()