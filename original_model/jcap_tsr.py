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


def table_structure_recognition(img):

    #print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # thresholding the image to a binary image
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    img_median = cv2.medianBlur(img_bin, 3)
  
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height*2)) #shape (kernel_len, 1) inverted! xD
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 9)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
  
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print("x", x, "y", y, "w", w, "h", h)
        x = x  # (-) shift towards left, (+) shift towards right 
        y = y-2 # (-) shift towards up, (+) shift towards down 
        h = h # expand height
        w = w # expand width
        # if (w < 0.9*img_width and h < 0.9*img_height):
        if (w < img_width and h < img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

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

    # calculating maximum number of cells
    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        #print("len",len(row[i]))
        if current > countcol:
            countcol = current
            index = i


    # Retrieving the center of each column
    #center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    #print("center",center)

    center = np.array(center)
    center.sort()

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

    return finalboxes, bitnot


def output_to_csv(finalboxes, img, ocr_model):
  img_thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
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
                  x = x  # (-) shift towards left, (+) shift towards right 
                  y = y  # (-) shift towards up, (+) shift towards down 
                  h = h  # expand height
                  w = w  # expand width
                  final_crop = img[x:x + h, y:y + w]
                  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                  # border = cv2.copyMakeBorder(finalimg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255])
                  resized = cv2.resize(final_crop, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                  # threshed = cv2.threshold(resized, 180, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)[1]
                  # img_enhance = cv2.detailEnhance(resized, sigma_s=5, sigma_r=0.15)                
                  # erosion = cv2.threshold(resized, 125, 255, cv2.THRESH_TOZERO)[1]
                  # erosion = cv2.erode(resized, kernel, iterations=1)
                  # dilation = cv2.dilate(erosion, kernel, iterations=1)

                  final_img = resized
                  
                  if(final_img.sum() != final_img.shape[0]*final_img.shape[1]*255):
                    # print('inside if')
                    out = ocr_model.readtext(final_img, detail = 0)
                    # if len(out) != 0:
                    #   pass
                  else:
                    out = ""
                  
                  if(out == ""):
                    out = ocr_model.readtext(final_img, detail = 0)
                    if len(out) != 0:
                      if(len(out[:-2]) >1):
                        out = ""

                  if len(out) != 0:
                    outlist = ','.join(out)
                    inner = inner + " " +  outlist #out[0] #out[:-2]
              outer.append(inner)

  # Creating a dataframe of the generated OCR list
  arr = np.array(outer)
  # table_dataframe = pd.DataFrame(arr.reshape(len(finalboxes), len(finalboxes[0])))
  # return table_dataframe
  return arr


def remove_extra_char(x):
  x = str(x).replace(" ", "")
  x = str(x).replace(',','.')
  return str(x)

def get_column_names(x):
  x = str(x).lower()
  x = str(x).replace(' ', '')
  x= str(x).replace('o', '0')
  x = str(x).replace('8', 's')
  x = str(x).replace('5', 's')
  x = str(x).replace('_', '')
  x = str(x).replace(',', '.')
  x = str(x).replace('m', 'n')
  return x

def dataframe_post_process(dataframe):
  dataframe1 = dataframe.copy()
  dataframe1 = dataframe1.replace('', np.NaN)
  dataframe1 = dataframe1.dropna(axis=0,thresh=3) 
  dataframe1 = dataframe1.dropna(axis=1,thresh=3)
  dataframe1.columns = dataframe1.iloc[0]
  dataframe1 = dataframe1[1:]
  dataframe1.columns = list(map(get_column_names, dataframe1.columns))

  num_rows = len(dataframe1.iloc[1:, :])

  for x in dataframe1:
    if str(x).startswith('ra'):
      dataframe1.rename(columns = {x:'ra'}, inplace=True)
    elif str(x).startswith('el'):
      dataframe1.rename(columns = {x:'el'}, inplace=True)
    elif str(x).startswith('rval'):
      dataframe1.rename(columns = {x:'n90'}, inplace=True)

  if 'ys' in dataframe1:
    ys = list(dataframe1['ys'])[1:] 
  else:
    ys = ['']*num_rows

  if 'uts' in dataframe1:
    uts = list(dataframe1['uts'])[1:] 
  else:
    uts = ['']*num_rows

  if 'el' in dataframe1:  
    el = list(dataframe1['el'])[1:]
  else:
    el = ['']*num_rows

  if 'ra' in dataframe1: 
    ra = list(dataframe1['ra'])[1:]
  else:
    ra = ['']*num_rows

  temp1 = [ys, uts, el, ra]

  temp2 = []
  for x in range(len(temp1)):
    temp = list(map(remove_extra_char, temp1[x]))
    temp2.append(temp)  
  
  temp4 = []
  for i, x in enumerate(temp2):
    temp3 = [x[z] for z in range(2,len(x),3)]
    temp4.append(temp3)
  
  temp4 = np.array(temp4).T.tolist()

  final_list = []
  for x in range(len(temp4)):
    temp5 = list(map(remove_extra_char, temp4[x]))
    temp6 = dict(zip(["ys","uts","el_perc","ra"], temp5))
    final_list.append(temp6)

  return final_list


def main(image, table_bbox, ocr_model, image_path):
    
    outpath='data/JCAP/csv'
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_ext='png'

    print('Cropping Table')
    x1, y1, x2, y2 = table_bbox
    # Crop table
    crop_img_table = image[int(y1):int(y2) , int(x1):int(x2)] 

    # Crop the table (for jsw)
    print('Cropping Again')
    crop_img_table = crop_img_table[:, 5:-10]
    
    # Deskew the table again
    print('Deskewing Table Image')
    angle, deskew_crop_table_img = correct_skew(crop_img_table)

    # Post Process Image for TSR
    print('Line Removal and post processing')
    deskew_crop_table_img = cv2.cvtColor(deskew_crop_table_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(deskew_crop_table_img, 255,1,1, 13, 20)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.medianBlur(thresh, 1)
    img_line_removed = lines_removal(thresh)    
    
    final_img = img_line_removed.copy()

    print('TSRing the table image')
    finalboxes, bitnot = table_structure_recognition(final_img)

 
    # Visualize TSR Results
    for x1 in finalboxes:
        for y1 in x1:
            for z1 in y1:
                x,y,w,h = z1  
                cv2.rectangle(final_img, (x, y), (x + w, y + h), (0,0,255), 2)

    # Save predicted table structure image
    tsr_outpath = outpath + str(img_name) + '_tsr.' + str(img_ext)
    print('tsr_outpath :', tsr_outpath)
    cv2.imwrite(tsr_outpath, final_img)

    # TABLE OCR 
    print('OCRing the table image')
    main_arr = output_to_csv(finalboxes, img_line_removed, ocr_model)
    
    # DF Post Process
    # print('Post Processing the dataframe')
    # final_list = dataframe_post_process(table_dataframe)

    return main_arr