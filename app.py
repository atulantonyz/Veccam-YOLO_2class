# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
 
st.write("""
         # MosquitoNet Classification - Jan 23
         """
         )

device=torch.device("cpu")

@st.cache(allow_output_mutation=True)
def load_model():
  model= torch.load('model/nonparallelmodel.pt', map_location = 'cpu')
  model = model.to(device)
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

basicTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])    



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

    st.image(image_disp, use_column_width= False)
    label, score = upload_predict(image, model)
    image_class = label #'mosquito' #str(predictions[0][0][1])
    st.write("### The image is classified as",species_all[image_class])
    st.write(f"#### The similarity score is approximately : {score*100:.2f} % ")
    # if score < 0.8:
    #     st.write("##### Confidence is below 80% - please upload a clearer image")
    print("The image is classified as ",image_class) #, "with a similarity score of",score)