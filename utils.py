from PIL import Image
import os


def halve_image(img, output_path):
    height = 256  # Desired final height
    width = 256  # Desired final width
    k = 0
    subdir = {0: '/truth/', 1: '/input/'}

    im = Image.open(img)
    imgwidth, imgheight = im.size
    assert (imgwidth == 512)
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            a = im.crop(box)
            path = output_path + subdir[k]
            os.makedirs(path, exist_ok=True)
            a.save(path + os.path.split(img)[-1])
            k += 1
    return "Done with all images"