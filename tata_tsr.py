## Importing Dependencies 
import pandas as pd
import json
import itertools
import numpy as np
import cv2
import os
from scipy.ndimage import interpolation as inter
from image_postprocess import correct_skew, lines_removal
import easyocr
from itertools import chain
def table_structure_recognition(img):
    #tess.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    #print("img_height", img_height, "img_width", img_width)

    # thresholding the image to a binary image
    # thresh, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    # inverting the image
    img_bin = 255 - img_bin
    # Plotting the image to see the output

    # countcol(width) of kernel as 100th of total width
    # kernel_len = np.array(img).shape[1] // 100
    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))  # shape (kernel_len, 1) inverted! xD
    #print("ver", ver_kernel)
    #print(ver_kernel.shape)

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  # shape (1,kernel_ken) xD
    #print("hor", hor_kernel)
    #print(hor_kernel.shape)

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #print(kernel)
    #print(kernel.shape)

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    # Plot the generated image

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
    #cv2.imwrite("/Users/marius/Desktop/img_vh.jpg", img_vh)
  
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    # Plotting the generated image
    # cv2_imshow(bitnot)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    #print(contours[0])
    #print(len(contours[0]))
    #print(cv2.boundingRect(contours[0]))

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)


    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    #print("lencontours", len(contours))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        print("x", x, "y", y, "w", w, "h", h)
        if (w < 0.9*img_width and h < 0.9*img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # cv2_imshow(image)

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

    #print("len box", len(box))
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    #print(column)
    #print(row)

    # calculating maximum number of cells
    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        print("len",len(row[i]))
        if current > countcol:
            countcol = current
            index = i

    #print("countcol", countcol)

    # Retrieving the center of each column
    #center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    #print("center",center)

    center = np.array(center)
    center.sort()
    #print("center.sort()", center)
    # Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin


def output_to_csv(finalboxes, img, ocr_model):
  # img_thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
  # cv2_imshow(img_thresh)
  testx = 0
  # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
  outer = []
  for i in range(len(finalboxes)):
      for j in range(len(finalboxes[i])):
          inner = ''
          if (len(finalboxes[i][j]) == 0):
              outer.append(' ')
          else:
              for k in range(len(finalboxes[i][j])):
                  y, x, w, h = int(finalboxes[i][j][k][0]), int(finalboxes[i][j][k][1]), int(finalboxes[i][j][k][2]), int(finalboxes[i][j][k][3])
                  # print('finalboxes after :', (y,x,w,h))
                  x = x  # (-) shift towards left, (+) shift towards right 
                  y = y  # (-) shift towards up, (+) shift towards down 
                  h = h  # expand height
                  w = w  # expand width
                  final_crop = img[x:x + h, y:y + w]
                  # print('final_crop shape :',final_crop.shape)
                  h,w,c = final_crop.shape
                  if h>1 or w>1: 
                    # cv2_imshow(final_crop)
                    # print(final_crop.shape)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    # border = cv2.copyMakeBorder(finalimg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255])
                    # INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4 
                    resized = cv2.resize(final_crop, None, fx=2.3, fy=2.3, interpolation=cv2.INTER_CUBIC)

                    # threshed = cv2.threshold(resized, 180, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)[1]
                    # img_enhance = cv2.detailEnhance(resized, sigma_s=5, sigma_r=0.15)
                    
                    # erosion = cv2.threshold(resized, 125, 255, cv2.THRESH_TOZERO)[1]
                    
                    # erosion = cv2.erode(resized, kernel, iterations=1)
                    # dilation = cv2.dilate(erosion, kernel, iterations=1)
                    final_img = resized
                    # cv2_imshow(final_img)
                    
                    if(final_img.sum() != final_img.shape[0]*final_img.shape[1]*255):
                      print('inside if')
                      out = ocr_model.readtext(final_img, detail = 0)
                      # out = pytesseract.image_to_string(final_img, config='')
                      print('out :', out)
                      # if len(out) != 0:
                      #   # out = ocr_model.readtext(final_img, detail = 0)
                      #   # out = pytesseract.image_to_string(final_img, config='')
                      #   print('out len :', len(out))
                      #   print('text :', out[0])

                    else:
                      print('inside else')
                      out = ""
                      print('text :',out)
                    
                    if(out == ""):
                      out = ocr_model.readtext(final_img, detail = 0)
                      # out = pytesseract.image_to_string(final_img, config='--psm 8')
                      # print('out :', out)
                      # print('out type: ', type(out))
                      # if len(out) != 0:
                        # out = ocr_model.readtext(final_img, detail = 0)
                        # out = pytesseract.image_to_string(final_img, config='--psm 8')
                      if(len(out[:-2]) >1):
                        out = ""
                        print('inside if if')
                        print('out len :', len(out))
                        print('text :', out[0])

                    if len(out) != 0:
                      outlist = ','.join(out)
                      inner = inner + " " +  outlist #out[0] #out[:-2]
                    # print('--------------------')
              outer.append(inner)

  # Creating a dataframe of the generated OCR list
  arr = np.array(outer)
  # result_dataframe = pd.DataFrame(arr.reshape(len(finalboxes), len(finalboxes[0])))
  # # print(dataframe)
  # data = result_dataframe.style.set_properties(align="left")
  # return result_dataframe, data
  return arr



def remove_extra_char(x):
  x = str(x).lower()
  x = str(x).replace(" ", "")
  x = str(x).replace(",", ".")
  return str(x)

def get_column_names(x):
  x = str(x).lower()
  x = str(x).replace(' ', '')
  return x


def dataframe_post_process(result_dataframe):
  result_dataframe1 = result_dataframe.copy()
  result_dataframe1 = result_dataframe1.replace('', np.NaN)
  result_dataframe1 = result_dataframe1.replace(' ', np.NaN)
  result_dataframe2 = result_dataframe1.dropna(axis=0,how='all').dropna(axis=1,how='all')
  result_dataframe3 = result_dataframe2.replace(np.NaN, '')
  result_dataframe4 = result_dataframe3.apply(np.vectorize(remove_extra_char))
  result_dataframe4.columns = result_dataframe4.iloc[0]
  result_dataframe5 = result_dataframe4[1:]
  result_dataframe5.columns = list(map(get_column_names, result_dataframe5.columns))
  num_rows = len(result_dataframe5.iloc[3:, :])

  for x in result_dataframe5.columns:
    if x.startswith('n'):
      result_dataframe5.rename(columns = {x:'n_90'}, inplace=True)

  if 'ys' in result_dataframe5:
    ys = list(result_dataframe5['ys'])[3:]
  else:
    ys = ['']*num_rows

  if 'uts' in result_dataframe5:
    uts = list(result_dataframe5['uts'])[3:] 
  else:
    uts = ['']*num_rows

  if 'el' in result_dataframe5:  
    el = list(result_dataframe5['el'])[3:]
  else:
    el = ['']*num_rows

  if 'ra' in result_dataframe5: 
    ra = list(result_dataframe5['ra'])[3:]
  else:
    ra = ['']*num_rows

  ## r90 is usually besides n90
  if 'n_90' in result_dataframe5: 
    n90 = list(result_dataframe5['n_90'])[3:]
    n90_loc = result_dataframe5.columns.get_loc("n_90")
    r90_loc = n90_loc + 1
    r90 = result_dataframe5.iloc[:, r90_loc].tolist()[3:]

  else:
    n90 = ['']*num_rows
    r90 = ['']*num_rows

  final_list = list(np.array([ys, uts, el, n90, r90, ra]).T)

  final_dict = {}
  final_result = []
  for x in range(len(final_list)):
    # print('final_list :', final_list[x])
    temp = list(map(get_column_names, final_list[x]))
    final_dict = dict(zip(["ys","uts","el_perc","n_90","r_90","ra"], temp))
    final_result.append(final_dict)

  return final_result


def main(image, table_bbox, ocr_model):
    print('Cropping Table')
    x1, y1, x2, y2 = table_bbox
    # Crop table
    crop_img_table = image[int(y1):int(y2) , int(x1):int(x2)]
    
    # Deskew the table again
    print('Deskewing Table Image')
    angle, deskew_crop_img_table = correct_skew(crop_img_table)
    
    # Pre-process Image
    thresh = cv2.threshold(deskew_crop_img_table, 150, 255, cv2.THRESH_BINARY)[1]
 
    print('TSRing the table image')
    finalboxes, bitnot = table_structure_recognition(thresh)
 
    # Visualize TSR Results
    # for x1 in finalboxes:
    #     for y1 in x1:
    #         for z1 in y1:
    #             x,y,w,h = z1  
    #             cv2.rectangle(final_img, (x, y), (x + w, y + h), (0,0,255), 2)

    # Save predicted table structure image
    # tsr_outpath = outpath + str(img_name) + '_tsr.' + str(img_ext)
    # print('tsr_outpath :', tsr_outpath)
    # cv2.imwrite(tsr_outpath, final_img)

    # TABLE OCR 
    print('OCRing the table image')
    main_arr = output_to_csv(finalboxes, deskew_crop_img_table, ocr_model)
    
    # # DF Post Process
    # print('Post Processing the dataframe')
    # final_list = dataframe_post_process(table_dataframe)

    return main_arr
