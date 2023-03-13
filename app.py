# %%writefile app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
# import mysql.connector
 
st.write("""
         # MosquitoNet Classification V4
         ### Hierarchical classification of mosquitos
         """
         )

device=torch.device("cpu")

@st.cache(allow_output_mutation=True)
def load_genus_model():
  model= torch.load('model/genus_model.pt', map_location = 'cpu')
  model = model.to(device)
  return model

@st.cache(allow_output_mutation=True)
def load_species_model():
  model= torch.load('model/steph_onevsall.pt', map_location = 'cpu')
  model = model.to(device)
  return model

with st.spinner('Model is being loaded..'):
  genus_model=load_genus_model()
  species_model=load_species_model()
 
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

genus_all = ["Anopheles","Culex","Other"]
 
species_all = ["Not Stephensi", "Stephensi"]



# def save_to_database(image_file, label, probab):
#     # Connect to the database
#     print('trying to insert into database')
#     connection = mysql.connector.connect(
#         host=st.secrets["host"],
#         user=st.secrets["user"],
#         password=st.secrets["password"],
#         database=st.secrets["database"]
#     )

#     cursor = connection.cursor()
#     print('connected to database')
#     # Insert the data into the table
#     insert_command = f"INSERT INTO `app_runs` (`id`, `imagename`, `prediction`, `probability`, `datetime`) VALUES ('', '{image_file}', '{label}', '{probab}' , CURRENT_TIMESTAMP);"
    
#     cursor.execute(insert_command)

#     connection.commit()
#     print('inserted into database')
#     cursor.close()
#     connection.close()

def upload_predict(upload_image, model):
    inputs = preprocess_image(upload_image)
    img_tensor = inputs.unsqueeze(0)

    # Run the model
    output = model(img_tensor)
    
    # get softmax of output

    output = F.softmax(output, dim=1)

    probab, pred = torch.max(output, 1)
    # print(output, pred, probab, probab.item())
    pred_class = pred.item()
    probab_value = probab.item()
    return pred_class, probab_value

if file is None:
    st.text("""Please upload an image file!""")
else:
    image = Image.open(file)

    # Open the image
    image_disp = image.copy()

    # Resize the image
    max_size = (400, 400)
    image_disp.thumbnail(max_size)

    st.image(image_disp, use_column_width= False)
    genus_index, genus_score = upload_predict(image, genus_model)
    # save_to_database(file.name, species_all[image_class], score*100)
    st.write("### The image is classified into Genus : ",genus_all[genus_index])
    st.write(f"#### The similarity score is approximately : {genus_score*100:.2f} % ")

    if genus_index == 0:
      species_index, species_score = upload_predict(image, species_model)
      st.write("### The image is classified into Species : ",species_all[species_index])
      st.write(f"#### The similarity score is approximately : {species_score*100:.2f} % ")
