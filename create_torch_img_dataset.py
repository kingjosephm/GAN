import os
import random
import math
import shutil

if __name__ == '__main__':

    rootDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_gray_thermal'
    outputDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_gray_thermal_pytorch'
    test_share = 0.001

    random.seed(123)

    # Check if output directory exists and delete if so (assuming script re-run)
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
        os.mkdir(outputDir)
        os.mkdir(outputDir + '/train')
        os.mkdir(outputDir + '/test')

    files = [i for i in os.listdir(rootDir) if 'png' in i]

    n_test = math.ceil(len(files) * test_share)
    n_train = len(files) - n_test

    test_files = random.sample(files, n_test)
    train_files = [i for i in files if i not in test_files]

    assert (len(test_files) == n_test) and (len(train_files) == n_train)

    all_files = {}
    all_files['train'] = train_files
    all_files['test'] = test_files

    # Copy and output files
    for key in all_files.keys():
        for file in all_files[key]:
            shutil.copy(src=os.path.join(rootDir, file),
                        dst=os.path.join(outputDir, key, file))