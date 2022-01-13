import os
import cv2

if __name__ == '__main__':

    path = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_gray_thermal'
    output = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_separated'
    thermal_left = True  # thermal image on left side, visible on right

    images = [i for i in os.listdir(path) if "png" in i or "jpg" in i]

    for img in images:

        image = cv2.cvtColor(cv2.imread(os.path.join(path, img)), cv2.COLOR_BGR2GRAY)
        w = image.shape[1] // 2

        # Slice concatenated images
        if thermal_left:
            therm = image[:, :w]
            vis = image[:, w:]
        else:
            therm = image[:, w:]
            vis = image[:, :w]

        # Output
        therm_output = os.path.join(output, 'therm')
        os.makedirs(therm_output, exist_ok=True)
        vis_output = os.path.join(output, 'vis')
        os.makedirs(vis_output, exist_ok=True)

        cv2.imwrite(os.path.join(therm_output, img), therm)
        cv2.imwrite(os.path.join(vis_output, img), vis)