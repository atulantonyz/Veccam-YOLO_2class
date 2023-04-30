# VecCAM Yolo_2_class + OCR
In this project, a mini web-based app was created using Streamlit that allows users to upload and analyze images.

## YOLO Model
A two-class YOLO model was utilized to get the cropped specimen ID and mosquito images. To prevent any important part of the ID from being cropped out, the width of the specimen ID bounding box was increased.

## easyOCR Library
The easyOCR library was used to extract the alphanumeric string associated with the specimen ID. A multi-step approach was implemented to ensure that the correct alphanumeric string is read and extracted. The OCR process first reads for characters, extracting all available letters within the image. Next, the process reads for digits, capturing any numbers associated with the alphanumeric string. Finally, these two sets of extracted data are combined to form the final, accurate alphanumeric string associated with the specimen ID.
