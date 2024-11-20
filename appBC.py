# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from util_functions import pad_image_to_square
import cv2
import easyocr
from ultralytics import YOLO
 
st.write("""
         # Ekyaalo - web app
         """
         )

device=torch.device("cpu")

@st.cache(allow_output_mutation=True)
def load_model():
  """
  Load PyTorch model from disk and move it to the appropriate device.

  Returns:
      model (torch.nn.Module): The loaded PyTorch model.
  """
  model= torch.load('model/best_mal_detect.pt', map_location = 'cpu')
  model = model.to(device)
  return model

@st.cache(allow_output_mutation=True)
def load_yolo():
  """
  Loads a custom YOLOv5 model from a local path and sends it to the CPU.

  Returns:
      yolo: A TorchHub model object representing the YOLOv5 model.
  """
  #yolo = torch.hub.load('ultralytics/yolov8', 'custom', path='model/best_cluster_detection.pt', force_reload=True)
  yolo=YOLO('model/best_cluster_detection.pt')
  yolo.to('cpu')
  return yolo

with st.spinner('Model is being loaded..'):
  model=load_model()
 
file10x = st.file_uploader("Upload the 10x image for cluster detection", type=["jpg", "png"])
file40x = st.file_uploader("Upload the 20x image for malignancy classification", type=["jpg", "png"])

mag = st.selectbox(
    "What is the magnification of the image ?",
    ("10x", "20x", "40x"),
)


basicTrans = transforms.Compose([ 
                                transforms.Resize([384,384]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])    
                                

def center_crop(image,img_sz):
    right,left,top,bottom=0,0,0,0  #padding needed on each edge
    width,height=image.size
    if width<img_sz:
        right = (img_sz-width)//2
        left = (img_sz-width)//2
    if height<img_sz:
        top=(img_sz-height)//2
        bottom=(img_sz-height)//2
    
    
    width, height = image.size 
    
    new_width = width + right + left 
    new_height = height + top + bottom 
    
    img_pad = Image.new(image.mode, (new_width, new_height), (0, 0, 0)) 
    
    img_pad.paste(image, (left, top)) 
    pad_w,pad_h=img_pad.size
    center_x,center_y=pad_w/2,pad_h/2
    left = center_x-img_sz//2
    top = center_y-img_sz//2
    right = center_x+img_sz//2
    bottom = center_y+img_sz//2
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = img_pad.crop((left, top, right, bottom))    
    return im1



def yolo_crop(image):#0 for specimen ID and 1 for mosquito
    """Apply YOLO object detection on an image and crop it around the detected mosquito.

    Args:
        image (PIL.Image.Image): Input image to crop.

    Returns:
        PIL.Image.Image: Cropped image centered around the detected mosquito.

    Raises:
        TypeError: If the input image is not a PIL image.

    Note:
        This function requires the `load_yolo` function to be defined and available in the current namespace.
        The YOLO model used by `load_yolo` must be able to detect mosquitoes in the input image.
    """

    yolo = load_yolo()
    img_c=center_crop(image,img_sz=1920)
    results = yolo.predict(img_c,conf=0.5,iou=0.7)
    boxes=[]
    try: 
       # crop the image
        for result in results[0]:
            xmin = int(result.boxes.xyxy.numpy()[0][0])
            ymin = int(result.boxes.xyxy.numpy()[0][1])
            xmax = int(result.boxes.xyxy.numpy()[0][2])
            ymax = int(result.boxes.xyxy.numpy()[0][3])
            conf0=result.boxes.conf.numpy()[0]
            boxes.append((xmin,ymin,xmax,ymax,conf0))

    except:
       st.write("No clusters detected")
    st.write("{} clusters detected".format(len(boxes)))   
    return boxes



def preprocess_image(image):
    image = basicTrans(image)
    return image

species_all = ["Non-malignant","Malignant"]

def upload_predict(upload_image, model):
    """
    Perform image classification on a given image using a pre-trained model.

    Args:
    - upload_image: A PIL Image object representing the image to be classified.
    - model: A PyTorch model object that has been trained on image classification.

    Returns:
    - pred_class: An integer representing the predicted class label of the image.
    - probab_value: A float representing the predicted class probability of the image.
    """
    inputs = preprocess_image(upload_image)
    img_tensor = inputs.unsqueeze(0)

    # Run the model
    output = model(img_tensor)
    
    # get softmax of output
    
    output = F.softmax(output, dim=1)

    probab, pred = torch.max(output, 1)
    print(output, pred, probab, probab.item())
    pred_class = pred.item()
    probab_value = probab.item()
    return pred_class, probab_value

# Main code block

if file10x is None:
    st.text(" ###### Please upload a 10x image file!")
else:
    image = Image.open(file10x)
    crop_pad_img=center_crop(image,1920)
    #image = image.resize((640,640))   #input size YOLO model was trained on
    # Open the image
    image_disp = crop_pad_img.copy()

    # Resize the image
    #max_size = (400, 400)
    #image_disp.thumbnail(max_size)
    st.write("### Uploaded Image")
    #st.image(image_disp, use_column_width= False)

    ### YOLO CROP
    img1 = ImageDraw.Draw(image_disp) 
    clusters = yolo_crop(image)
    for cluster in clusters:
        xy=cluster[:4]
        conf=cluster[4]
        print(xy)
        img1.rectangle(xy, fill=None, outline='blue', width=5)
        img1.text((xy[0], xy[1]), "{:.2f}".format(conf), fill=(0, 0, 255))
    
    max_size = (400, 400)
    image_disp.thumbnail(max_size)
    st.write("### Shape of the image is", image.size)
    st.write("### Shape of the crop and pad image is", crop_pad_img.size)
    st.image(image_disp, use_column_width= False)
   
    
    
if file40x is None:
    st.text(" ###### Please upload a 20x image file!")
else:
    ### PAD IMAGE
    image40xpnga = Image.open(file40x)
    image40x = Image.new("RGB", image40xpnga.size, (255, 255, 255))
    image40x.paste(image40xpnga) # 3 is the alpha channel
    #image40x=np.array(image40x)
    #print(image40x.shape)
    #image40x=image40x[:,:,:3]
    #Choose size of center crop based on magnification
    cc_size=(784 if mag=="40x" else 192 if mag=="10x" else 384)
    image40x_c=center_crop(image40x,img_sz=cc_size)
    st.write("### Cropped and Padded Image")
    image_disp = image40x_c.copy()
    max_size = (400, 400)
    image_disp.thumbnail(max_size)
    st.write("### Shape of the image is", image40x.size)
    st.write("### Shape of the crop and pad image is", image40x_c.size)
    st.image(image_disp, use_column_width= False)

    ### CLASSIFY
    label, score = upload_predict(image40x_c, model)
    image_class = label 
    st.write("### The image is classified as",species_all[image_class])
    st.write(f"#### The confidence score is approximately : {score*100:.2f} % ")
