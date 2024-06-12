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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    # inverting the image
    img_bin_inv = 255 - img_bin
    # countcol(width) of kernel as 100th of total width
    #kernel_len = np.array(img).shape[1] // 100
    kernel_len_ver = max(10,img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) #shape (kernel_len, 1) inverted
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) #shape (1,kernel_ken)

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    
    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
 
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.dilate(img_vh, kernel, iterations=5)

    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY ))

    bitor = cv2.bitwise_or(img_bin, img_vh)

    img_median = cv2.medianBlur(bitor, 3)

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height*2)) #shape (kernel_len, 1) inverted! xD
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 1)) #shape (kernel_len, 1) inverted! xD
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
    # Plotting the generated image

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
        if (w < 0.9*img_width and h < 0.9*img_height):
            x = x  # (-) shift towards left, (+) shift towards right 
            y = y - 5 # (-) shift towards up, (+) shift towards down 
            h = h*1.2 # expand height
            w = w*1 # expand width
            image = cv2.rectangle(img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)
            box.append([int(x), int(y), int(w), int(h)])

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
        if current > countcol:
            countcol = current
            index = i

    # Retrieving the center of each column
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]

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
                  y = y # (-) shift towards up, (+) shift towards down 
                  h = h + 7  # expand height
                  w = w + 1  # expand width
                  final_crop = img[x:x + h, y:y + w]
                  h,w,c = final_crop.shape
                  if h==0 or w==0:
                    inner = ''
                  else:
                    resized = cv2.resize(final_crop, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
                    final_img = resized
                  
                    if(final_img.sum() != final_img.shape[0]*final_img.shape[1]*255):
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
                      inner = inner + " " +  outlist 
              outer.append(inner)

  # Creating a dataframe of the generated OCR list
  arr = np.array(outer)
  # table_dataframe = pd.DataFrame(arr.reshape(len(finalboxes), len(finalboxes[0])))
  return arr


## DataFrame Post Process
def get_column_names(x):
  x = str(x).lower()
  x = str(x).replace(' ', '')
  x= str(x).replace('o', '0')
  x = str(x).replace('_', '')
  x = str(x).replace(',', '.')
  x = str(x).replace('m', 'n')
  return x

def dataframe_post_process(dataframe):
  dataframe1 = dataframe.copy()
  final_dict = {}
  final_result = []
  coil_count = 0 
  dataframe1 = dataframe1.replace('', np.NaN)
  dataframe1 = dataframe1.dropna(axis=0,thresh=5)  
  dataframe1 = dataframe1.dropna(axis=1,thresh=3)
  dataframe1.columns = dataframe1.iloc[0]
  dataframe1 = dataframe1[1:]
  dataframe1.columns = list(map(get_column_names, dataframe1.columns))
  for x in dataframe1:
    if x.startswith('ra'):
      dataframe1.rename(columns = {x:'ra'}, inplace=True)
    elif x.startswith('el'):
      dataframe1.rename(columns = {x:'el'}, inplace=True)
    elif x.startswith('r'):
      dataframe1.rename(columns = {x:'r90'}, inplace=True)

  num_rows = len(dataframe1.iloc[3:, :])

  if 'ys' in dataframe1:
    ys = list(dataframe1['ys'])[3:]
  else:
    ys = ['']*num_rows
    
  if 'uts' in dataframe1:
    uts = list(dataframe1['uts'])[3:]
  else:
    uts = ['']*num_rows
  
  if 'el' in dataframe1:
    el = list(dataframe1['el'])[3:]
  else:
    el = ['']*num_rows

  if 'r90' in dataframe1:
    r90 = list(dataframe1['r90'])[3:]
  else:
    r90 = ['']*num_rows

  if 'n90' in dataframe1:
    n90 = list(dataframe1['n90'])[3:]
  else:
    n90 = ['']*num_rows
  
  if 'ra' in dataframe1:
    ra = list(dataframe1['ra'])[3:]
  else:
    ra = ['']*num_rows

  final_list = list(np.array([ys, uts, el, r90, n90, ra]).T)

  for x in range(len(final_list)):
    temp = list(map(get_column_names, final_list[x]))
    final_dict = dict(zip(["ys","uts","el_perc","r_90","n_90","ra"], temp))
    final_result.append(final_dict)
  return final_result



def main(image, table_bbox, ocr_model):
    x1, y1, x2, y2 = table_bbox
    # Crop table
    img_table = image[int(y1):int(y2) , int(x1):int(x2)]   
    
    # Deskew the table again
    angle, img_table = correct_skew(img_table)
    
    # Post Process Image for TSR
    img_line_removed = lines_removal(img_table)    
    thresh = cv2.threshold(img_line_removed, 180, 255, cv2.THRESH_BINARY)[1]
    
    # Crop the table (for jsw)
    crop_thresh = thresh[10:-10, 10:]
    final_img = crop_thresh.copy()

    finalboxes, bitnot = table_structure_recognition(final_img)
 
    # Visualize TSR Results
    # for x1 in finalboxes:
    #     for y1 in x1:
    #         for z1 in y1:
    #             x,y,w,h = z1  
    #             cv2.rectangle(final_img, (x, y), (x + w, y + h), (0,0,255), 2)

    # TABLE OCR 
    main_arr = output_to_csv(finalboxes, crop_thresh, ocr_model)

    # # DF Post Process
    # final_list = dataframe_post_process(table_dataframe)

    return main_arr