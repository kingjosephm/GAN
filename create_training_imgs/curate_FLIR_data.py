import os
import json
import cv2
import imutils
import numpy as np

"""

    credit: 
        https://medium.com/@aswinvb/how-to-perform-thermal-to-visible-image-registration-c18a34894866
        
"""

def align_images(therm: np.ndarray, vis: np.ndarray, output: str, dims: tuple = (512, 640)):
    """
    Aligns (1024, 1224) visible image onto (512, 640) thermal image. Thermal image is subset of visible, so this
        function identifies overlapping pixels and restricts to these.
    :param therm: np.ndarray, thermal image
    :param vis: np.ndarray, matched visible image
    :param output: str, full path and name of file to output
    :param dims: tuple, (height, width) dimensions of desired output image
    :return:
        None
        Writes (as default) 512 x 1280 concatenated image to disk
    """

    therm = cv2.cvtColor(therm, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

    # Apply Canny line sequence detection algorithm
    therm_canny = cv2.Canny(therm, 100, 200)

    # bookkeeping variable to keep track of the matched region
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:

        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(vis, width=int(vis.shape[1] * scale))
        r = vis.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < therm.shape[0] or resized.shape[1] < therm.shape[1]:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 100, 200)
        result = cv2.matchTemplate(edged, therm_canny, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + therm.shape[1]) * r), int((maxLoc[1] + therm.shape[0]) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(vis, (startX, startY), (endX, endY), (0, 0, 255), 2)
    crop_img = vis[startY:endY, startX:endX]

    # resize both images to desired dimensions
    resized_therm = cv2.resize(therm, (dims[1], dims[0]))
    resized_vis = cv2.resize(crop_img, (dims[1], dims[0]))

    # Concat horizontally
    concatenated = cv2.hconcat([resized_therm, resized_vis])

    cv2.imwrite(output, concatenated)


if __name__ == '__main__':

    output_dir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_rgb_thermal'

    ################################
    ###### Images from Europe ######
    ################################
    europe = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_ADAS_DATASET/Europe_1_0_2_full/Europe_1_0_2'

    for subset in ['val', 'train']:

        # Crosswalk
        with open(os.path.join(europe, subset, 'europe_thermal_to_rgb.json')) as f:
            crosswalk = json.load(f)
        crosswalk = crosswalk['thermal_to_rgb_ordered_frames']  # converts to list

        for matched_pair in range(len(crosswalk)):

            # See if matched images exist, read in if they do
            therm_path = os.path.join(europe, subset, 'thermal_8_bit', crosswalk[matched_pair]['thermal_filename'])
            vis_path = os.path.join(europe, subset, 'RGB', crosswalk[matched_pair]['rgb_filename'])

            if (os.path.exists(therm_path)) and (os.path.exists(vis_path)):
                therm = cv2.imread(therm_path)
                vis = cv2.imread(vis_path)

                if (isinstance(therm, np.ndarray)) and (isinstance(vis, np.ndarray)):  # ensure they read in correctly

                    name = f"europe_{subset}_{matched_pair}.png"

                    align_images(therm=therm, vis=vis, output=os.path.join(output_dir, name))

            else:

                if subset == 'val':  # check if image in val_video folder instead

                    # See if matched images exist, read in if they do
                    therm_path = os.path.join(europe, 'val_video', 'thermal_8_bit', crosswalk[matched_pair]['thermal_filename'])
                    vis_path = os.path.join(europe, 'val_video', 'RGB', crosswalk[matched_pair]['rgb_filename'])

                    if (os.path.exists(therm_path)) and (os.path.exists(vis_path)):
                        therm = cv2.imread(therm_path)
                        vis = cv2.imread(vis_path)

                        if (isinstance(therm, np.ndarray)) and (isinstance(vis, np.ndarray)):  # ensure they read in correctly

                            name = f"europe_video_{matched_pair}.png"

                            align_images(therm=therm, vis=vis, output=os.path.join(output_dir, name))

    ################################
    ###### Images from Europe ######
    ################################
    sf = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_ADAS_DATASET/FLIR_ADAS_SF_1_0_0'

    for subset in ['val', 'train']:

        files = [i for i in os.listdir(os.path.join(sf, subset, 'thermal_8_bit'))]

        for file in files:

            # See if matched images exist, read in if they do
            therm_path = os.path.join(sf, subset, 'thermal_8_bit', file)
            vis_path = os.path.join(sf, subset, 'RGB', file)

            if (os.path.exists(therm_path)) and (os.path.exists(vis_path)):
                therm = cv2.imread(therm_path)
                vis = cv2.imread(vis_path)

                if (isinstance(therm, np.ndarray)) and (isinstance(vis, np.ndarray)):  # ensure they read in correctly

                    name = f"sf_{subset}_{file[5:-5]}.png"

                    align_images(therm=therm, vis=vis, output=os.path.join(output_dir, name))

    ##### video #####

    files = [i for i in os.listdir(os.path.join(sf, 'video', 'thermal_8_bit'))]

    for file in files:

        # See if matched images exist, read in if they do
        therm_path = os.path.join(sf, 'video', 'thermal_8_bit', file)
        vis_path = os.path.join(sf, 'video', 'RGB', file.replace('jpeg', 'jpg'))  # RGB files use JPG suffix

        if (os.path.exists(therm_path)) and (os.path.exists(vis_path)):
            therm = cv2.imread(therm_path)
            vis = cv2.imread(vis_path)

            if (isinstance(therm, np.ndarray)) and (isinstance(vis, np.ndarray)):  # ensure they read in correctly

                name = f"sf_video_{file[5:-5]}.png"

                align_images(therm=therm, vis=vis, output=os.path.join(output_dir, name))