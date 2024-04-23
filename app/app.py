# Importing essential libraries and modules
from markupsafe import Markup
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from utils.disease import disease_dic

import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9


disease_classes = [
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   ]

disease_model_path = 'models/resnet9-mdlsd.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()







def predict_image(img, model=disease_model):
   
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


app = Flask(__name__)
@ app.route('/')
def home():
    title = 'LeafGuard- Home'
    return render_template('index.html', title=title)
@app.route('/disease-predict', methods=['GET','POST'])
def disease_prediction():
    title = 'LeafGuard - Disease Detection'
    if request.method == 'POST':
        print(request.files)
        file = request.files['files']
       
        try:
            img = file.read()
            print("the image is:-",img)
            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)