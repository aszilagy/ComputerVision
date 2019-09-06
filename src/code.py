import cv2
import math
import numpy as np
import os
import sys
from objloader_simple import *
import operator

MIN_MATCHES = 200

def main():
    """
    This method handles calls to all other methods. 
    It gathers objects (and converts them to OBJ format) and model reference images
    The frame handling is also done in openCV as well as ORB and BFMatcher
    Once the model is chosen, projection_matrix() and render() are called accordingly

    The program can be run from the main directory by running:
        python src/code.py
    """
    print(len(sys.argv))
    if len(sys.argv) == 5:
        fu = sys.argv[1]
        fv = sys.argv[2]
        u0 = sys.argv[3]
        v0 = sys.argv[4]

    else:
        fu, fv, u0, v0 = 760, 760, 360, 360
        print("WARNING: You have not specified any parameters. Using Default")
        print("Please use the following format:  python src/code.py <fu> <fv> <u0> <v0>")
        print("<fu, fv> represents the focal length and <u0, v0> represents the camera parameters")
        print("These values default to <760, 760> and <360, 360> respectively")

    # Camera Parameters = [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
    camera_parameters = np.array([[fu, 0, u0], [0, fv, v0], [0, 0, 1]])

    print("The camera parameters chosen are:")
    print("Focal Length <fu, fv>: <",fu,fv,">")
    print("Camera Values <u0, v0>: <",u0,v0,">")

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    dir_name = os.getcwd()
    model_co2 = cv2.imread(os.path.join(dir_name, 'reference/co2.jpg'), 0)
    model_mol = cv2.imread(os.path.join(dir_name, 'reference/mol.jpg'), 0)
    model_pi = cv2.imread(os.path.join(dir_name, 'reference/pi.jpg'), 0)
    model_pc = cv2.imread(os.path.join(dir_name, 'reference/laptop.jpg'), 0)

    kp_model_co2, des_model_co2 = orb.detectAndCompute(model_co2, None)
    kp_model_mol, des_model_mol = orb.detectAndCompute(model_mol, None)
    kp_model_pi, des_model_pi = orb.detectAndCompute(model_pi, None)
    kp_model_pc, des_model_pc = orb.detectAndCompute(model_pc, None)

    obj_co2 = OBJ(os.path.join(dir_name, 'models/co2.obj'), swapyz=True)  
    obj_mol = OBJ(os.path.join(dir_name, 'models/mol.obj'), swapyz=True)  
    obj_pi = OBJ(os.path.join(dir_name, 'models/pi.obj'), swapyz=True)  
    obj_pc = OBJ(os.path.join(dir_name, 'models/laptop.obj'), swapyz=True)  

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        if type(des_frame) != type(None):
            matches_co2 = bf.match(des_model_co2, des_frame)
            matches_mol = bf.match(des_model_mol, des_frame)
            matches_pi = bf.match(des_model_pi, des_frame)
            #Removed Temporarily for Demo purposes
            #matches_pc = bf.match(des_model_pc, des_frame)
            matches_pc = []

        else:
            #print("0 matches found:",type(des_frame))
            matches_co2, matches_mol, matches_pi, matches_pc = [], [], [], []

        tmp_dict = {'co2':len(matches_co2),'mol':len(matches_mol),'pi':len(matches_pi),'pc':len(matches_pc)}
        max_val = max(tmp_dict.items(), key=operator.itemgetter(1))[0]

        if max_val == 'co2':
            matches, model = matches_co2, model_co2
            kp_model, des_model = kp_model_co2, des_model_co2
            obj = obj_co2
            scale_var = 3
            #print("Found a Co2: %s " % len(matches_co2))

        elif max_val == 'mol':
            matches, model = matches_mol, model_mol
            kp_model, des_model = kp_model_mol, des_model_mol
            obj = obj_mol
            scale_var = 3
            #print("Found a Mol: %s " % len(matches_mol))

        elif max_val == 'pi':
            matches, model = matches_pi, model_pi
            kp_model, des_model = kp_model_pi, des_model_pi
            obj = obj_pi
            scale_var = 3
            #print("Found a Pi: %s " % len(matches_pi))

        elif max_val == 'pc':
            matches, model = matches_pc, model_pc
            kp_model, des_model = kp_model_pc, des_model_pc
            obj = obj_pc
            scale_var = 50
            #print("Found a Pyr: %s " % len(matches_pc))

        else:
            #print("UNKNOWN: Something went wrong!")
            pass

        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = model.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography)
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            if len(matches) > MIN_MATCHES:
                projection = projection_matrix(camera_parameters, homography)  
                frame = render(frame, obj, projection, model, scale_var)
            #matches arg
            frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            if key & 0xFF == ord('h'):
                print("Ensure you are in the correct directory /Project/")
                print("Please use the following format:  python src/code.py <fu> <fv> <u0> <v0>")
                print("<fu, fv> represents the focal length and <u0, v0> represents the camera parameters")
                print("These values default to <760, 760> and <360, 360> respectively")
                print("Pressing q will quit out of the program. Pressing h will bring up the help menu")


    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, scale=3):
    """
    Renders the 3D image onto the frame utilizing the projection matrix previously calculated
    This function uses the (obj, model) pairing and the projection matrix

    Params:
      img.....Image frame to be modified
      obj.....Object (OBJ Class) associated with Image
      proj....Projection Matrix returned by projection_matrix()
      model...2D Reference image of the Object
      scale...Wanted scale to be displayed of the 3D obj
    Returns:
      img...Final image frame with rendered object displayed
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = model.shape

    for face in obj.faces:
        points = np.array([vertices[vertex - 1] for vertex in face[0]])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        #B,G,R
        cv2.fillConvexPoly(img, imgpts, (0,0,0))

    return img

def projection_matrix(camera_parameters, homography):
    """
    Computes the projection matrix utilizing camera parameters and estimated homography matrix

    Params:
      camera_parameters...Estimated Camera Parameters (defined in beginning)
      homography..........Estimated Homography Matrix 
                (Obtained from cv2.findHomography and utilizing RANSAC)
    Returns:
      proj_matr...The projection matrix obatined through calculations
    """
    homography = homography * (-1)
    g_mat = np.dot(np.linalg.inv(camera_parameters), homography)
    lin_alg = math.sqrt(np.linalg.norm(g_mat[:,0], 2) * np.linalg.norm(g_mat[:,1], 2))

    translation = g_mat[:,2]/lin_alg

    c = (g_mat[:,0]/lin_alg) + (g_mat[:,1]/lin_alg)
    p = np.cross((g_mat[:,0]/lin_alg),(g_mat[:,1]/lin_alg))
    d = np.cross(c, p)

    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    proj_matr = np.dot(camera_parameters, projection)
    return proj_matr

if __name__ == '__main__':
    main()
