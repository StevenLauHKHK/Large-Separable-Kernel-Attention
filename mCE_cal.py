import csv
import numpy as np

alexnet_error = {
    'gaussian_noise': 88.6,
    'shot_noise': 89.4,
    'impulse_noise': 92.3,
    'defocus_blur': 82.0,
    'glass_blur': 82.6,
    'motion_blur': 78.6,
    'zoom_blur': 79.8,
    'snow': 86.7,
    'frost': 82.7,
    'fog': 81.9,
    'brightness': 56.5,
    'contrast': 85.3,
    'elastic_transform': 64.6,
    'pixelate': 71.8,
    'jpeg_compression': 60.7,
    'speckle_noise': 84.5,
    'gaussian_blur': 78.7,
    'spatter': 71.8,
    'saturate': 65.8
}

model = 'lsk_w_dilation'
model_complex = 'van_base'
ce_list = {}
with open('results-all.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if len(row) > 0 and model == row[-4] and model_complex == row[-5]:
            top1_acc = float(row[0])
            ce = 100 - top1_acc
            distortion_type = row[-2]
            distortion_level = int(row[-1])
            if distortion_type not in ce_list:
                ce_list[distortion_type] = [ce]
            else:
                ce_list[distortion_type].append(ce)

mean_acc = 0
for k, v in ce_list.items():
    acc = 100-np.mean(v)
    print(k, acc)
    mean_acc += acc
mean_acc = mean_acc / 19
print('mean acc=',mean_acc)

final_mean_ce = 0
for k, v in ce_list.items():
    mean_ce = np.mean(v)
    mean_ce = mean_ce / alexnet_error[k]
    ce_list[k] = mean_ce
    final_mean_ce += mean_ce
final_mean_ce = final_mean_ce / 19 * 100
print('mCE by distortion type=', ce_list)
print('mCE=', final_mean_ce)





