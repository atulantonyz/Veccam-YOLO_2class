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
         # VectorCAM webapp 08/23
         """
         )

device=torch.device("cpu")

@st.cache_resource
def load_model():
  """
  Load PyTorch model from disk and move it to the appropriate device.

  Returns:
      model (torch.nn.Module): The loaded PyTorch model.
  """
  model= torch.load('models/best_species_fold5_08_23.pt', map_location = 'cpu')
  model=model.module
  model = model.to(device)
  return model
  
@st.cache_resource
def load_abd_model():
  """
  Load PyTorch model from disk and move it to the appropriate device.

  Returns:
      model (torch.nn.Module): The loaded PyTorch model.
  """
  model= torch.load('models/best_abdomen_fold3_08_23.pt', map_location = 'cpu')
  model=model.module
  model = model.to(device)
  return model
  
@st.cache_resource
def load_sex_model():
  """
  Load PyTorch model from disk and move it to the appropriate device.

  Returns:
      model (torch.nn.Module): The loaded PyTorch model.
  """
  model= torch.load('models/best_sex_fold3_08_23.pt', map_location = 'cpu')
  model=model.module
  model = model.to(device)
  return model  
  

@st.cache_resource
def load_yolo():
  """
  Loads a custom YOLOv5 model from a local path and sends it to the CPU.

  Returns:
      yolo: A TorchHub model object representing the YOLOv5 model.
  """
  yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='models/YOLO_08_23.pt', force_reload=True)
  yolo.to('cpu')
  return yolo

with st.spinner('Model is being loaded..'):
  model=load_model()
  abd_model=load_abd_model()
  sex_model=load_sex_model()
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


basicTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])    

def yolo_crop(image):
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
    results = yolo(image)
    try: 
       # crop the image
        xmin = int(results.xyxy[0].numpy()[0][0])
        ymin = int(results.xyxy[0].numpy()[0][1])
        xmax = int(results.xyxy[0].numpy()[0][2])
        ymax = int(results.xyxy[0].numpy()[0][3])
        conf0=results.xyxy[0].numpy()[0][4]
        class0=results.xyxy[0].numpy()[0][-1]
        im_crop = image.crop((xmin, ymin, xmax , ymax))
        print("Image cropped successfully!")
        print('Genus',class0)
        return class0,conf0,im_crop

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
                "Aedes",
                "Mansonia",
                "Non-mosquito"]
abdomen_all=["Unfed","Fully Fed","Gravid"]
sex_all=["Female","Male"]                

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
    abd_output=abd_model(img_tensor)
    sex_output=sex_model(img_tensor)
    # get softmax of output

    output = F.softmax(output, dim=1)
    abd_output = F.softmax(abd_output, dim=1)
    sex_output = F.softmax(sex_output, dim=1)
    #species
    probab, pred = torch.max(output, 1)
    print(output, pred, probab, probab.item())
    pred_class = pred.item()
    probab_value = probab.item()
    #abdomen
    probab1, pred1 = torch.max(abd_output, 1)
    print(abd_output, pred1, probab1, probab1.item())
    pred_class1 = pred1.item()
    probab_value1 = probab1.item()
    #sex
    probab2, pred2 = torch.max(sex_output, 1)
    print(sex_output, pred2, probab2, probab2.item())
    pred_class2 = pred2.item()
    probab_value2 = probab2.item()
    
    return pred_class, probab_value,pred_class1, probab_value1,pred_class2, probab_value2

# Main code block

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
    genus,conf,yolo_cropped_image = yolo_crop(image)
    #st.write("### Cropped Image")
    # st.write shape of image
    st.write("### Shape of the cropped image is", yolo_cropped_image.size)

    ### PAD IMAGE
    image = pad_image_to_square(yolo_cropped_image)
    st.write("### Cropped and Padded Image")
    image_disp = image.copy()
    image_disp.thumbnail(max_size)
    st.image(image_disp, use_column_width= False)
 
    ### CLASSIFY
    label, score,label1, score1,label2, score2 = upload_predict(image, model)
    st.write("### Species: ",species_all[label])
    st.write(f"#### Confidence : {score*100:.2f} % ")
    st.write("### Sex: ",sex_all[label2])
    st.write(f"#### Confidence : {score2*100:.2f} % ")
    st.write("### Abdomen: ",abdomen_all[label1])
    st.write(f"####  Confidence : {score1*100:.2f} % ")
