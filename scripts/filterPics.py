# from PIL import Image
# import os
# n = [6,7,8,9]
# out_dir = r"D:\data\hackathon\wiki_crop\filtered"
# for i in n:
#     dirname = r"D:\data\hackathon\wiki_crop\0" + str(i)
#     dir = os.listdir(dirname)
#
#     for file in dir:
#         with Image.open(dirname + '/' + file) as f:
#             width, height = f.size
#             if width < 256 or height < 256:
#                 continue
#             else:
#                 f.save(out_dir + '/' + file)

import cv2
import os
in_path = r"D:\data\hackathon\wiki_crop\filtered2"
out_path = r"D:\data\hackathon\wiki_crop\filtered3"
dir = os.listdir(in_path)
for filename in dir:
    im = cv2.imread(in_path + '/' + filename)
    small = cv2.resize(im, (256, 256))
    # cv2.imshow("", small)
    cv2.imwrite(out_path + '/' + filename, small)
    # cv2.waitKey(0)

