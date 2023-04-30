# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from util_functions import pad_image_to_square
import cv2
import easyocr
 
st.write("""
         # MosquitoNet Classification - V8 -  Yolo (zoomed out) Included 
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
  model= torch.jit.load('models/species_03_16_23.ptl', map_location = 'cpu')
  model = model.to(device)
  return model

@st.cache(allow_output_mutation=True)
def load_yolo():
  """
  Loads a custom YOLOv5 model from a local path and sends it to the CPU.

  Returns:
      yolo: A TorchHub model object representing the YOLOv5 model.
  """
  yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='models/YOLO_wlabel.pt', force_reload=True)
  yolo.to('cpu')
  return yolo

with st.spinner('Model is being loaded..'):
  model=load_model()
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


basicTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])    

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
    results = yolo(image)
    try: 
       # crop the image
        xmin0 = int(results.xyxy[0].numpy()[0][0])
        ymin0 = int(results.xyxy[0].numpy()[0][1])
        xmax0 = int(results.xyxy[0].numpy()[0][2])
        ymax0 = int(results.xyxy[0].numpy()[0][3])
        class0=results.xyxy[0].numpy()[0][-1]
        xmin1 = int(results.xyxy[0].numpy()[1][0])
        ymin1 = int(results.xyxy[0].numpy()[1][1])
        xmax1 = int(results.xyxy[0].numpy()[1][2])
        ymax1 = int(results.xyxy[0].numpy()[1][3])
        class1=results.xyxy[0].numpy()[1][-1]
        #im_crop = image.crop((ymin, xmin, ymax , xmax))
        if class0==0.0:
            im0_crop = image.crop((xmin0-10, ymin0, xmax0+10 , ymax0))  # increasing width for label bounding box for crop
            im1_crop = image.crop((xmin1, ymin1, xmax1 , ymax1))
        else:
            im0_crop = image.crop((xmin0, ymin0, xmax0 , ymax0))
            im1_crop = image.crop((xmin1-10, ymin1, xmax1+10 , ymax1))  # increasing width for label bounding box for crop
            
        print("Image cropped successfully!")
        if class0==0.0:
            return im0_crop,im1_crop
        else:    
            return im1_crop,im0_crop

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

    # output = F.softmax(output, dim=1)

    probab, pred = torch.max(output, 1)
    print(output, pred, probab, probab.item())
    pred_class = pred.item()
    probab_value = probab.item()
    return pred_class, probab_value

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
    yolo_cropped_label,yolo_cropped_image = yolo_crop(image)
    st.write("### Cropped Image")
    # st.write shape of image
    st.write("### Shape of the image is", yolo_cropped_label.size)
    
    # Convert the image to a numpy array
    img_array = np.array(yolo_cropped_label.resize((int(yolo_cropped_label.size[0]/4),int(yolo_cropped_label.size[1]/4))))# smaller size might be helpful for OCR

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Use the reader to extract text from the image
    result=reader.readtext(img_array)
    if(len(result)==0):
        st.write(" specimen ID not detected")
    else:    
        result_ch = reader.readtext(img_array)
        result_num = reader.readtext(img_array,allowlist='0123456789')
        chr_str=result_ch[0][1][:3]         #read first three letters
        chr_num=result_num[0][1][-3:]       #read last three digits
        ID_str=chr_str+chr_num
        st.write(ID_str)

    # Print the extracted text using Streamlit
    #if(len(result)==0):
    #    st.write(" specimen ID not detected")
    #else:    
    #    st.write(ID_str)

    ID_image = yolo_cropped_label
    st.write("### Specimen ID")
    image_disp = ID_image.copy()
    image_disp.thumbnail(max_size)
    st.image(image_disp, use_column_width= False)
    
  
    #st.write("### Cropped Image")
    # st.write shape of image
    st.write("### Shape of the image is", yolo_cropped_image.size)
    
    ### PAD IMAGE
    image = pad_image_to_square(yolo_cropped_image)
    st.write("### Cropped and Padded Image")
    image_disp = image.copy()
    image_disp.thumbnail(max_size)
    st.image(image_disp, use_column_width= False)

    ### CLASSIFY
    label, score = upload_predict(image, model)
    image_class = label 
    st.write("### The image is classified as",species_all[image_class])
    st.write(f"#### The similarity score is approximately : {score*100:.2f} % ")
