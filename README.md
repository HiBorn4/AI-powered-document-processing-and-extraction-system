data 1

YS
UTS
EL
Ra

For EL the Prompts which we used further are considering GL values due to confusion if we use a new chat then we can get good accuracy

PDF 1

PDF 2

PDF 3

PDF 4

PDF 5


data 2
YS UTS EL rBAR  Ra n 

n value incorrect
ys uts n
n
n
n uts ys


data 3

YS UTS EL RRVALUE RaMICRO NVALUE


data 4

second stage

data 5

YP
TS
EL
Ra

May 27th Requirement
One Prompt  
Data Unavailable 
High Accuracy for Mapping
Refine the Prompts

May 28th Results
All the values are being mapped wrong
Reason:
On one side we are tell openai to:
1. Make it DATA UNAVAILABLE  if a particular attribute is not present
2. On the other hand we are telling it to Rigorously search for each and every attribute
    2.1 Because No attribute has DATA UNAVAILABLE it is rigorously mapping any trash value***
3. We are facing EL GL cross mapping due to combined prompt
4. It is not recognizing the Coil No. too in other invoices we have Dimensions



Advantages
High Accuracy 
Robust

May 28th New ML Model

ResNet50
Yolov8

Step 1: Convert PDFs to images.

Step 2: Tabular Data Detection using YOLOv5 & Resnet50:

Using YOLOv5, detect the tabular data, crop the image, and convert it to PDF
Using Resnet50, detect the tabular data, crop the image, and convert it to PDF
Compare YOLOv5 accuracy with Resnet50 accuracy
Compare Original image inputs with YOLOv5 results and Resnet50 results

Step 3: Apply necessary image preprocessing steps.

Step 4: Using Adobe Acrobat OCR, extract the excel format of the cropped converted PDF.

Step 5: Using python code, remove the noise from the extracted excel file and maintain the general structure of the output.



ResNet50
Labelling
Classes
Train 5 Layers 168 neurons each layer
Freezing each layer
adjusting weights according to the output

Yolov8
Yolov8 High Accuracy nearly 92% accuracte
High amount of dataset is required nearly 100


May29th 
Explain the Model
Make specified information regarding the documents
Switch Cases for specific Company Format
Each Cases ke liye example 