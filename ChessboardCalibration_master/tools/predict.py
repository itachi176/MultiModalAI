import pickle 
import cv2
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from calibration.Functions import distance, roundToPoint5
from calibration import Predictor, UndistortMethodTester
from regression import Regression
import sys
import argparse
def parse_args():
    parser =argparse.ArgumentParser(description="Run calibration")
    parser.add_argument('--image', type=str, help='path to image', default= './calibration/chessboard.jpg')
    parser.add_argument('--calib_data', type=str, help='calibration data', default='./calibration/calib.pkl')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--out_dir', type=str, help='output images path', default='./out_dir/images/')
    parser.add_argument('--undistort', type = bool, help='validate with traditional camera calibration methods', default=False)
    parser.add_argument('--distortedImagePath', type=str, help='dataset', default='./dataset_29072021/')
    parser.add_argument('--regression_data', help="regression data", type =str, default='./regression/regression_data.npy')
    args = parser.parse_args()
    return args

def pred(x, y, img):

    calib_data = "./ChessboardCalibration_master/calibration/calib.pkl"
    origin = (615,24)
    out_dir = './ChessboardCalibration_master/out_dir/images/'
    predictor = Predictor(img_path=img, calib_data = calib_data, origin=origin, out_dir=out_dir)
    regression_data = './ChessboardCalibration_master/regression/data.csv'
    # if not args.undistort:

    regressor = Regression(regression_data)
    
    real_x, real_y = predictor.predict(x, y)
    reg_x, reg_y = regressor.predict(real_x,real_y)
    print("real:", real_x, real_y)
    print("regress:", reg_x, reg_y)
    # elif args.undistort:
    #     umt = UndistortMethodTester(args)
    #     umt.undistort()
        #corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        #print(corners_df)
    return reg_x, reg_y
# if __name__ == "__main__":
#     pred()