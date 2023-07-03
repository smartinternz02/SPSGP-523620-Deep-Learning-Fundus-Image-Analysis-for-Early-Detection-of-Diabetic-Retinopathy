import os

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# Load the Diabetic Retinopathy model

model = models.resnet101(pretrained=True)
model.classifier=nn.Linear(1024,102)
model.load_state_dict(torch.load('stage-1.pth', map_location='cpu'), strict=False)
model.eval()


# Define the image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## Define classes for Diabetic Retinopathy classification
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the image from the request
        image = request.files['image']
        img_filename = image.filename
        # Save the uploaded image
        image.save(os.path.join(app.root_path, 'static/images', img_filename))

        # Preprocess the image
        img = Image.open(image)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        # Perform inference
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            # Assuming `output` is your tensor with the model's output

            # Apply softmax normalization


        if len(output) == 0:
            return render_template('error.html', message='Inference error')
        probabilities = F.softmax(output, dim=1)

        # Get the predicted class index
        _, predicted = torch.max(probabilities, 1)

        # Normalize the predicted value to the range of 0 to 4
        normalized_prediction = predicted.item() / (probabilities.shape[1] - 1) * 4

        # Round the normalized prediction to the nearest integer
        prediction = round(normalized_prediction)

        # Ensure the prediction is within the range of 0 to 4
        prediction = max(0, min(4, prediction))


        if prediction >= len(classes):
            return render_template('error.html', message='Invalid prediction')

        pred = classes[prediction]

        return render_template('result.html', prediction=pred, img_filename=img_filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





