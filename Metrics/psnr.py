import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def psnr(ref_path, target_path):
    
    target = plt.imread(target_path)
    ref = plt.imread(ref_path)
    if(ref.shape[-1]==4):
      ref = ref[:,:,:3]
    if(target.shape[-1]==4):
      target = target[:,:,:3]
    if(ref.shape != target.shape):
      target = cv2.resize(target, ref.shape[:-1], interpolation=cv2.INTER_AREA)

    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)