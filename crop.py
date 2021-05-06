import numpy as np
import pandas as pd
import os
import fitz
from PIL import Image, ImageChops
from pdf2image import convert_from_path

import argparse
import math
from pathlib import Path
import cv2

"""
Modules to be installed:
pip install PyMuPDF
pip install fitz
"""


def main():
    pw = os.getcwd()
    files = os.listdir(pw + "\data")

    names_list = []
    images_path_list = []
    for file in files:
        file_path = f"{pw}\data\{file}"
        name = file.split("_")[1].split(".")[0]
        pdf_file = fitz.open(file_path)
        out_dir = f"{pw}\images\{name}"
        images_path = f"images\{name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # finding number of pages in the pdf
        number_of_pages = len(pdf_file)

        # iterating through each page in the pdf
        for current_page_index in range(number_of_pages):
            idx = number_of_pages
            # iterating through each image in every page of PDF
            for img_index, img in enumerate(pdf_file.getPageImageList(current_page_index)):
                xref = img[0]
                image = fitz.Pixmap(pdf_file, xref)
                print(image)
                # if it is a is GRAY or RGB image
                if image.n < 5:
                    image.writePNG("{}/image{}-{}-{}.png".format(out_dir, current_page_index, img_index, idx))
                    names_list.append(str(name))
                    images_path_list.append(f"{out_dir}/image{current_page_index}-{img_index}.png")
                # if it is CMYK: convert to RGB first
                else:
                    new_image = fitz.Pixmap(fitz.csRGB, image)
                    new_image.writePNG("{}/image{}-{}-{}.png".format(out_dir, current_page_index, img_index, idx))
                    names_list.append(str(name))
                    images_path_list.append(f"{out_dir}/image{current_page_index}-{img_index}.png")

                idx += 1

    df = pd.DataFrame(zip(names_list, images_path_list), columns=['names', 'filePath'])
    df.to_csv(pw + "/output.csv")


def crop_pdf():
    pw = os.getcwd()
    files = os.listdir(pw + "\data")
    poppler_path = pw + "/poppler/bin"

    names_list = []
    for i, f2 in enumerate(files):
        if ".pdf" in f2:
            file_path = f"{pw}\data\{f2}"
            name = f2.split("_")[1].split(".")[0]
            out_dir = f"{pw}\images\{name}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                images = convert_from_path(f2, poppler_path=poppler_path)
                for j in range(len(images)):
                    images[j].save(out_dir + '/' + str(i) + '_' + str(j) + '.jpg', 'JPEG')

    root_folder_path = pw + "\images"
    root_folders = os.listdir(root_folder_path)

    cropped_images_path = pw + "\cropped_images"
    if not os.path.exists(cropped_images_path):
        os.makedirs(cropped_images_path)

    names_list = []
    images_path_list = []
    for folder in root_folders:
        root_image = pw + f"/images/{folder}"
        for image in os.listdir(root_image):
            image_path = os.path.join(root_image, image)
            out_file_path = cropped_images_path + f"/{image}" 
            img = cv2.imread(image_path)
            height,width=img.shape[:2]
            start_row,start_col=120, 30
            end_row,end_col=620, 780
            cropped=img[start_row:end_row,start_col:end_col]
            cv2.imwrite(out_file_path,cropped)
            #cv2.imshow("Cropped_Image",cropped)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            names_list.append(str(folder))
            images_path_list.append(f"/cropped_images/{image}")

    df = pd.DataFrame(zip(names_list, images_path_list), columns=['names', 'filePath'])
    df.to_csv(pw + "/output.csv")


def more_crop():
    img = Image.open("pUq4x.png")
    pixels = img.load()

    print (f"original: {img.size[0]} x {img.size[1]}")
    xlist = []
    ylist = []
    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if pixels[x, y] != (255, 255, 255, 255):
                xlist.append(x)
                ylist.append(y)
    left = min(xlist)
    right = max(xlist)
    top = min(ylist)
    bottom = max(ylist)

    img = img.crop((left-10, top-10, right+10, bottom+10))
    img.show()




if __name__ == '__main__':
    # main()
    crop_pdf()
