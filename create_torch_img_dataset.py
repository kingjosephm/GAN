import os
import random
import math
import shutil

if __name__ == '__main__':

    rootDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_gray_thermal'
    outputDir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/FLIR_matched_gray_thermal_pytorch'
    val_share = 0.1
    test_share = 0.001

    random.seed(123)

    # Check if output directory exists and delete if so (assuming script re-run)
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
        os.mkdir(outputDir)
        os.mkdir(outputDir + '/train')
        os.mkdir(outputDir + '/val')
        os.mkdir(outputDir + '/test')

    files = [i for i in os.listdir(rootDir) if 'png' in i]

    n_val = math.ceil(len(files) * val_share)
    n_test = math.ceil(len(files) * test_share)
    n_train = len(files) - n_val - n_test

    val_files = random.sample(files, n_val)
    test_files = random.sample([i for i in files if i not in val_files], n_test)
    train_files = [i for i in files if i not in val_files and i not in test_files]

    assert (len(val_files) == n_val) and (len(test_files) == n_test) and (len(train_files) == n_train)

    all_files = {}
    all_files['train'] = train_files
    all_files['val'] = val_files
    all_files['test'] = test_files

    # Copy and output files
    for key in all_files.keys():
        for file in all_files[key]:
            shutil.copy(src=os.path.join(rootDir, file),
                        dst=os.path.join(outputDir, key, file))