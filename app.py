# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from util_functions import pad_image_to_square
import cv2
 
st.write("""
         # MosquitoNet Classification - V4.2 -  Yolo Testing 
         """
         )

device=torch.device("cpu")

@st.cache(allow_output_mutation=True)
def load_model():
  model= torch.jit.load('model/species_03_16_23.ptl', map_location = 'cpu')
  model = model.to(device)
  return model

@st.cache(allow_output_mutation=True)
def load_yolo():
  yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='model/yoloV5trained.pt', force_reload=True)
  return yolo

with st.spinner('Model is being loaded..'):
  model=load_model()
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


basicTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])    

def yolo_crop(image):
    yolo = load_yolo()
    results = yolo(image)
    try: 
       # crop the image
        x1 = int(results.xyxy[0].numpy()[0][0])
        y1 = int(results.xyxy[0].numpy()[0][1])
        x2 = int(results.xyxy[0].numpy()[0][2])
        y2 = int(results.xyxy[0].numpy()[0][3])

        im_crop = image.crop((x1, y1, x2, y2))
        print("Image cropped successfully!")
        return im_crop

    except:
       st.write("No mosquito detected")
    return image

def preprocess_image(image):
    image = basicTrans(image)
    return image
 
species_all = ["An. funestus",
                "An. gambiae",
                "An. other",
                "Culex",
                "Other"]

def upload_predict(upload_image, model):
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

    probabilities = model(inputs.unsqueeze(0).to(device))

    _, predicted = torch.max(probabilities.data, 1)
    pred_arr = predicted.cpu().detach().numpy()
    return species_all[pred_arr[0]]

if file is None:
    st.text(" ###### Please upload an image file!")
else:
    image = Image.open(file)

    # Open the image
    image_disp = image.copy()

    # Resize the image
    max_size = (400, 400)
    image_disp.thumbnail(max_size)
    st.write("### Uploaded Image")
    st.image(image_disp, use_column_width= False)

    ### YOLO CROP
    yolo_cropped_image = yolo_crop(image)
    st.write("### Cropped Image")
    # st.write shape of image
    st.write("### Shape of the image is", yolo_cropped_image.size)

    ### PAD IMAGE
    image = pad_image_to_square(yolo_cropped_image)
    st.write("### Padded Image")
    image_disp = image.copy()
    image_disp.thumbnail(max_size)
    st.image(image_disp, use_column_width= False)

    ### CLASSIFY
    label, score = upload_predict(image, model)
    image_class = label 
    st.write("### The image is classified as",species_all[image_class])
    st.write(f"#### The similarity score is approximately : {score*100:.2f} % ")
