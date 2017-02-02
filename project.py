import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from moviepy.editor import *

calib_params_file = "calibration_matrix.p"
test_image = "test_images/straight_lines1.jpg"
ym_per_pix = 30/720     # meters per pixel in y dimension
xm_per_pix = 3.7/700    # meters per pixel in x dimension
curv_thresh_max = 3000
curv_thresh_min = 200
stripes = 10
window_half_w = 50
min_pixels = 50
min_points = 5

mtx, dist, left, right = None, None, None, None


def calibration_matrix():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    return mtx, dist


def load_cal_matrix():
    if not os.path.exists(calib_params_file):
        mtx, dist = calibration_matrix()
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(calib_params_file, "wb"))
    else:
        dist_pickle = pickle.load(open(calib_params_file, "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    return mtx, dist


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def color_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return s_binary


def get_persp_transform():
    # define 4 source points for perspective transformation
    src = np.float32([[220, 719], [1220, 719], [750, 477], [553, 480]])
    # define 4 destination points for perspective transformation
    dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def show_images(*images,title=None):
    f, axes = plt.subplots(1,len(images), squeeze=False)
    f.set_size_inches((6 * len(images),8))
    if title and len(images)==1 and type(title) == str:
        title = [title]
    for i in range(len(images)):
        img = images[i]
        ax = axes[0][i]
        if title is not None:
            assert len(title) == len(images)
            t = title[i]
            ax.text(0.5, 1.05, t, transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='center')
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")


def show(bird_eye_view, left_fitx, lp_y, right_fitx, rp_y):
    f, axes = plt.subplots(1, 1, squeeze=False)
    f.set_size_inches((6, 8))
    img = bird_eye_view
    ax = axes[0][0]
    if len(img.shape) == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap="gray")
    ax.plot(left_fitx, lp_y, 'o', color='red', markersize=5)
    ax.plot(right_fitx, rp_y, 'o', color='blue', markersize=5)


def lane_points(bird_eye_view, split, h, window_half_w, window_h, prev_lx = None, prev_rx = None):
    lp_x = []
    lp_y = []
    rp_x = []
    rp_y = []

    histogram = np.sum(bird_eye_view[h // 2:, :], axis=0)
    left_lane_hist = histogram[:split]
    right_lane_hist = histogram[split:]

    if not prev_lx:
        peaks_left = ndimage.measurements.center_of_mass(left_lane_hist)
        prev_lx = int(peaks_left[0]) if peaks_left else None

    if not prev_rx:
        peaks_right = ndimage.measurements.center_of_mass(right_lane_hist)
        prev_rx = (split + int(peaks_right[0])) if peaks_right else None

    for bottom_y in range(h, 0, -window_h):

        win_top = bottom_y - window_h
        win_bottom = bottom_y

        if prev_lx:
            win_l_left = prev_lx - window_half_w
            win_l_right = prev_lx + window_half_w
            window_l = bird_eye_view[win_top:win_bottom, win_l_left:win_l_right]

            pix = np.sum(window_l)

            if pix > min_pixels:
                hist_left = np.sum(window_l, axis=0)
                peaks = ndimage.measurements.center_of_mass(hist_left)[0]
                if not np.isnan(peaks):
                    lp_x.append(win_l_left + int(peaks))
                    lp_y.append(bottom_y - window_h // 2)
                    prev_lx = win_l_left + int(peaks)

        if prev_rx:
            win_r_left = prev_rx - window_half_w
            win_r_right = prev_rx + window_half_w
            window_r = bird_eye_view[win_top:win_bottom, win_r_left:win_r_right]

            pix = np.sum(window_r)

            if pix > min_pixels:
                hist_right = np.sum(window_r, axis=0)
                peaks = ndimage.measurements.center_of_mass(hist_right)[0]
                if not np.isnan(peaks):
                    rp_x.append(win_r_left + int(peaks))
                    rp_y.append(bottom_y - window_h // 2)
                    prev_rx = win_r_left + int(peaks)

    return np.array(lp_x[::-1]), np.array(lp_y[::-1]), np.array(rp_x[::-1]), np.array(rp_y[::-1])


def x(coef1, coef2, coef3, y):
    return coef1 * y ** 2 + coef2 * y + coef3


def curv(coef1, coef2, y):
    return ((1 + (2 * coef1 * y * ym_per_pix + coef2) ** 2) ** 1.5) / np.absolute(2 * coef1)


def curvature(lp_x, lp_y, rp_x, rp_y, y_eval = 350):
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lp_y * ym_per_pix, lp_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(rp_y * ym_per_pix, rp_x * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = curv(left_fit_cr[0], left_fit_cr[1], y_eval)
    right_curverad = curv(right_fit_cr[0], right_fit_cr[1], y_eval)

    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, left_fit_cr, right_fit_cr


def distance(left_fit_cr, right_fit_cr, y_eval = 350):
    x_left = x(left_fit_cr[0], left_fit_cr[1], left_fit_cr[2], y_eval * ym_per_pix)
    x_right = x(right_fit_cr[0], right_fit_cr[1], right_fit_cr[2], y_eval * ym_per_pix)

    return np.absolute(x_left - x_right)


def lane_parallel(left_fit_cr, right_fit_cr, threshold = .3):
    y = np.linspace(0, 719, num=10)
    dist = distance(left_fit_cr, right_fit_cr, y_eval=y * ym_per_pix)
    min = np.min(dist)
    max = np.max(dist)

    return np.absolute(min - max) < threshold


def validate(c_l, c_r, left_cr, right_cr, dist, curv_thresh = 1000):
    # Checking that they have similar curvature
    if np.absolute(c_l - c_r) > curv_thresh and c_l < 4000 and c_r < 4000:
        return False

    # Checking that they are separated by approximately the right distance horizontally
    if dist < 3.4 or dist >= 4.3:
        return False

    # Checking that they are roughly parallel
    parallel = lane_parallel(left_cr, right_cr)
    if not parallel:
        return False

    return True


def print_stats(image, c_l, c_r, dista, diff):
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(image,
                'Estimated curvature: Left: {0}m, Right: {1}m.'.format(round(c_l, 2),
                                                                       round(c_r, 2)),
                (10, 50), font, 1, (0, 0, 255), 2)

    cv2.putText(image,
                'Estimated distance between lanes: {0}m'.format(round(dista, 2)),
                (10, 100), font, 1, (200, 200, 0), 2)

    cv2.putText(image, 'Offset from lane center : {0}m'.format(round(diff, 2)),
                (10, 150), font, 1, (0, 0, 0), 2)


def draw_lanes(Minv, bird_eye_view, image, left_fitx, lp_y, right_fitx, rp_y):
    warp_zero = np.zeros_like(bird_eye_view).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usableformat for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, np.array(lp_y, dtype=np.float64)]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, np.array(rp_y, dtype=np.float64)])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32(pts), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result


def offset(left_fit, right_fit):
    mid = image.shape[1] / 2
    y_val = 350
    to_left_line = mid - (left_fit[0] * y_val ** 2 + left_fit[1] * y_val + left_fit[2])
    to_right_line = right_fit[0] * y_val ** 2 + right_fit[1] * y_val + right_fit[2] - mid
    return to_right_line, to_left_line


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #x values for detected line pixels
        self.allx = []
        #y values for detected line pixels
        self.ally = []


def process_image(image):
    # undistorting an image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # thresholding
    col_binary = color_thresh(image, thresh=(170, 255))
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    combined_binary = np.zeros_like(col_binary)
    combined_binary[(col_binary == 1) | (gradx == 1)] = 1

    # persepctive transformation
    M, Minv = get_persp_transform()
    bird_eye_view = cv2.warpPerspective(combined_binary, M, (1280, 720), flags=cv2.INTER_LINEAR)

    # get origin lane points from the previous line lane
    prev_lx = left.allx[-1] if len(left.allx) > 0 and left.detected else None
    prev_rx = right.allx[-1] if len(right.allx) > 0 and right.detected else None

    # find points from birds eye view
    h, w = bird_eye_view.shape
    split = w // 2
    window_h = h // stripes
    lp_x, lp_y, rp_x, rp_y = lane_points(bird_eye_view, split, h, window_half_w, window_h,
                                         prev_lx, prev_rx)

    # check if any line has been detected. If not, use the valid line from previous frame
    if len(lp_x) <= min_points:
        left.detected = False
        lp_x = left.allx
        lp_y = left.ally

    if len(rp_x) <= min_points:
        right.detected = False
        rp_x = right.allx
        rp_y = right.ally

    if len(lp_x) == 0 or len(lp_y) == 0 or len(rp_x) == 0 or len(rp_y) == 0:
        return image

    # check curvature, distance between lines. If problem occurs, use both lines form the previous frame
    curv_l, curv_r, coef_l, coef_r = curvature(lp_x, lp_y, rp_x, rp_y)
    lanes_dist = distance(coef_l, coef_r)
    valid = validate(curv_l, curv_r, coef_l, coef_r, lanes_dist)
    if not valid:
        if len(left.allx) > 0 and len(right.allx) > 0:
            lp_x = left.allx
            lp_y = left.ally
            rp_x = right.allx
            rp_y = right.ally
        else:
            return image

    # fit the lines
    left_fit = np.polyfit(lp_y, lp_x, 2)
    right_fit = np.polyfit(rp_y, rp_x, 2)
    left_fitx = x(left_fit[0], left_fit[1], left_fit[2], lp_y)
    right_fitx = x(right_fit[0], right_fit[1], right_fit[2], rp_y)
    lastx_l = left_fit[2]
    lastx_r = right_fit[2]
    firstx_l = x(left_fit[0], left_fit[1], left_fit[2], h-1)
    firstx_r = x(right_fit[0], right_fit[1], right_fit[2], h-1)
    lp_y = np.hstack(([0], lp_y, [(h-1)]))
    rp_y = np.hstack(([0], rp_y, [(h-1)]))
    left_fitx = np.hstack(([lastx_l], left_fitx, [firstx_l]))
    right_fitx = np.hstack(([lastx_r], right_fitx, [firstx_r]))

    # calculate offset from the center
    to_right_line, to_left_line = offset(left_fit, right_fit)
    offs = (to_right_line - to_left_line) * xm_per_pix

    # print stats
    print_stats(image, curv_l, curv_r, lanes_dist, offs)

    # Create an image to draw the lines on
    result = draw_lanes(Minv, bird_eye_view, image, left_fitx, lp_y, right_fitx, rp_y)

    #show(bird_eye_view, left_fitx, lp_y, right_fitx, rp_y)
    #plt.waitforbuttonpress()

    # update state
    left.detected = True
    right.detected = True
    left.allx = left_fitx.astype(np.int64)
    left.ally = lp_y
    right.allx = right_fitx.astype(np.int64)
    right.ally = rp_y

    return result


if __name__ == "__main__":
    mtx, dist = load_cal_matrix()
    left = Line()
    right = Line()

    image = cv2.imread(test_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # output = process_image(image)
    # plt.imshow(output)
    # plt.waitforbuttonpress()

    white_output = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4", audio=False)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)