import pickle
import os
import cv2
import shutil
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from numpy import asarray
import tqdm
import builtins
import pandas as pd
import json
import time
from datasets import load_from_disk

seg_index = 0

# for ind in range(0,780):
#         #remove the true_positives and false_positives folders

#         shutil.rmtree("baseline/segment_files/"+str(ind)+"/true_positives")
#         shutil.rmtree("baseline/segment_files/"+str(ind)+"/false_positives")

# for root, dirs, files in os.walk("data/BUSI2Dtrain/labels", topdown=True):
#     for fil in sorted(files):
#         print(fil)

        # # convert png to np bool array
        # img = Image.open("data/BUSI2Dtrain/labels/"+str(fil))
        # print(img.size)
        # numpydata = asarray(img,dtype=bool)
        # print(numpydata.shape)
        # numpydata=np.transpose(numpydata) #np and PIL have different dimensions
        # print(numpydata.shape)


        #change filename so that it is in the form (b/m/n)000.png

        # found_num=False
        # digits=[]
        # firstchar=fil[0]
        # for char in str(fil):
        #     if char == '(':
        #         found_num=True
        #         continue
        #     if char == ')':
        #         break
        #     if found_num == True:
        #         digits+=char
        # if len(digits) == 1:
        #     os.rename("data/BUSI2Dtrain/labels/"+str(fil),"data/BUSI2Dtrain/labels/"+firstchar+"00"+digits[0]+".png")
        # elif len(digits) == 2:
        #     os.rename("data/BUSI2Dtrain/labels/"+str(fil),"data/BUSI2Dtrain/labels/"+firstchar+"0"+digits[0]+digits[1]+".png")
        # elif len(digits) == 3:
        #     os.rename("data/BUSI2Dtrain/labels/"+str(fil),"data/BUSI2Dtrain/labels/"+firstchar+digits[0]+digits[1]+digits[2]+".png")


        #remove everything currently in the false_negatives folder

        # for filename in os.listdir("baseline/segment_files/"+str(seg_index)+"/false_negatives"):
        #     print(filename)
        #     os.remove("baseline/segment_files/"+str(seg_index)+"/false_negatives/"+str(filename))
        # seg_index=seg_index+1

        #add img pkl to folder

        # with open(
        #     "baseline/segment_files/" + str(seg_index) + "/false_negatives/" + str(seg_index) + ".pkl",
        #     "wb",
        # ) as seg_pkl:
        #     pickle.dump(numpydata, seg_pkl, protocol=5)
        # seg_index = seg_index+1

#remove best_points files
# for i in range(0,647):
#     os.remove("baseline/best_points/"+str(i)+"/"+str(i)+".pkl")

# #below code is to show the pkls
s_color = {   
    # "true_positives": [255, 255, 255],
    # "false_positives": [128, 255, 128],
    "false_negatives": [255, 128, 128],
}

builtins.image_size = (256, 256)
ds_hf = load_from_disk("data/traindataset")
segment_files = "baseline/segment_files"
segment_stats = pd.read_csv("baseline/segment_stats.csv", index_col=0)
image_perf_df = pd.read_csv("baseline/image_perf_df.csv", index_col=0)
# with open("baseline/guide_clicks.json", "r") as gcfile:
#     guide_clicks = json.load(gcfile)
#     guide_clicks = {int(k): v for k, v in guide_clicks.items()}

for i in range(0,647):
    image_stats = segment_stats[segment_stats["image_index"] == i]
    original_image = Image.open("data/BUSIresized/BUSI2Doriginal/"+str(i)+".png")
    label_arr = asarray(Image.open("data/BUSIresized/BUSI2Dgts/"+str(i)+".png"))
    # if(i<437):
    #     original_image = Image.open("data/BUSIresized/BUSI2Doriginal/"+str(i)+".png")
    #     if(i<10):
    #         label_arr = asarray(Image.open("data/BUSIresized/BUSI2Dgts/"+str(i)+".png"))
    #     elif(i<100):
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/b0"+str(i)+".png"))
    #     else:
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/b"+str(i)+".png"))
    # elif(i<647):
    #     original_image = Image.open("data/BUSI2Dtrain/images/malignant ("+str(i-436)+").png")
    #     if(i-436<10):
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/m00"+str(i-436)+".png"))
    #     elif(i-436<100):
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/m0"+str(i-436)+".png"))
    #     else:
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/m"+str(i-436)+".png"))
    # else:
    #     original_image = Image.open("data/BUSI2Dtrain/images/normal ("+str(i-646)+").png")
    #     if(i-646<10):
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/n00"+str(i-646)+".png"))
    #     elif(i-646<100):
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/n0"+str(i-646)+".png"))
    #     else:
    #         label_arr = asarray(Image.open("data/BUSI2Dtrain/labels/n"+str(i-646)+".png"))
    # clicks = guide_clicks[i]
    label_img = ImageOps.colorize(
        Image.fromarray(label_arr, mode="L"),
        black=[0, 0, 0],
        white=[255, 255, 0],
        blackpoint=0,
        whitepoint=255,
    )
    img_label_merged = Image.composite(
        label_img, original_image, Image.fromarray(label_arr, mode="L")
    )

    s_merged = original_image
    for st in s_color.keys():
        st_stats = image_stats[image_stats["segment_type"] == st]
        for s in st_stats.iterrows():
            with open(
                segment_files
                + "/"
                + str(i)
                + "/"
                + st
                + "/"
                + str(i)
                + ".pkl",
                "rb",
            ) as s_file:
                s_arr = pickle.load(s_file)
                s_arr = s_arr * 255
                s_img = ImageOps.colorize(
                    Image.fromarray(s_arr, mode="L"),
                    black=[0, 0, 0],
                    white=s_color[st],
                    blackpoint=0,
                    whitepoint=255,
                )
                s_merged = Image.composite(
                    s_img, s_merged, Image.fromarray(s_arr, mode="L")
                )
                # print(s_arr)
                # print(s_img)
                # print(s_merged)

    # for c in clicks["clicks_positive_tp"]:
    #     # Pillow and Numpy disagree on the meaning of X and Y
    #     # we use the Numpy convention
    #     # so for Pillow we swap X and Y
    #     ImageDraw.Draw(s_merged).rectangle(
    #         xy=[(c[1] - 1, c[0] - 1), (c[1] + 1, c[0] + 1)],
    #         outline=(0, 255, 0),
    #         fill=(0, 255, 0),
    #         width=1,
    #     )

    # for c in clicks["clicks_positive"]:
    #     # Pillow and Numpy disagree on the meaning of X and Y
    #     # we use the Numpy convention
    #     # so for Pillow we swap X and Y
    #     ImageDraw.Draw(s_merged).rectangle(
    #         xy=[(c[1] - 1, c[0] - 1), (c[1] + 1, c[0] + 1)],
    #         outline=(0, 255, 0),
    #         fill=(0, 255, 0),
    #         width=1,
    #     )

    # for c in clicks["clicks_negative"]:
        # Pillow and Numpy disagree on the meaning of X and Y
        # we use the Numpy convention
        # so for Pillow we swap X and Y
    with open(
        "baseline/best_points"
        + "/"
        + str(i)
        + "/"
        + str(i)
        + ".pkl",
        "rb",
    ) as c_file:
        c = pickle.load(c_file)
    # c=[178, 263]
    print(c)
    for siz in range(0,len(c)):
        ImageDraw.Draw(s_merged).rectangle(
            xy=[(c[siz][1] - 1, c[siz][0] - 1), (c[siz][1] + 1, c[siz][0] + 1)],
            outline=(255, 0, 0),
            fill=(255, 0, 0),
            width=1,
        )

    #text
    oi_text = "image_index=" + str(i)
    oi_text += "\n" + "IoU=" + str(round(image_perf_df.loc[i, "iou"], ndigits=4)) 
    draw = ImageDraw.Draw(original_image)
    bbox = draw.multiline_textbbox((0, 0), oi_text)
    draw.rectangle(bbox, fill="black")
    draw.text((0, 0), oi_text)

    big_frame = Image.new(
        mode="RGB",
        size=(builtins.image_size[0] * 3, builtins.image_size[0]),
        color=(0, 0, 0),
    )
    big_frame.paste(original_image, box=(0, 0))
    big_frame.paste(img_label_merged, box=(builtins.image_size[0], 0))
    big_frame.paste(s_merged, box=(builtins.image_size[0] * 2, 0))
    # big_frame.show(cv2.waitKey(5000))
    big_frame.save("resized_point_images/"+str(i)+".png")
    # time.sleep(10)
    #display(big_frame)