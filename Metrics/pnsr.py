import numpy as np
import math
import matplotplib.pyplot as plt

def psnr(ref_path, target_path):
    
    target = plt.imread(target_path)
    ref = plt.imread(ref_path)

    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)