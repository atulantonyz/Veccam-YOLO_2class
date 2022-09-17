# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
 


 
st.write("""
         # Image Classification
         """
         )

@st.cache(allow_output_mutation=True)
def load_model():
  model= torch.load('model/Species.pt')
  if model:
    st.write(""" # Got it""")
  else:
    st.write(""" # Got it""")
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

mosquito_transforms = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize([300,300]),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device=torch.device("cpu")

def load_image(image):
    image = mosquito_transforms(image)
    
    return image
 
species_all = ["An. funestus",
                "An. gambiae",
                "An. other",
                "Culex",
                "Other",
                "An. stephensi"]

def upload_predict(upload_image, model):
    inputs = load_image(upload_image)
    probabilities = model(inputs.unsqueeze(0).to(device))

    _, predicted = torch.max(probabilities.data, 1)
    pred_arr = predicted.cpu().detach().numpy()
    return species_all[pred_arr[0]]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    label = upload_predict(image, model)
    image_class = label #'mosquito' #str(predictions[0][0][1])
    score= 0.9 #np.round(predictions[0][0][2]) 
    st.write("The image is classified as",image_class)
    # st.write("The similarity score is approximately",score)
    print("The image is classified as ",image_class, "with a similarity score of",score)