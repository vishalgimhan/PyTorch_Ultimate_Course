# %% Packages
from flask import Flask, request
from model_class import MultiClassNet
import torch
import json

# %% model instance
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
local_file_path = 'model_iris.pt'
model.load_state_dict(torch.load(local_file_path))


app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Please send a POST request with input data for prediction."
    elif request.method == 'POST':
        data = request.data.decode('utf-8')
        dict_data = json.loads(data.replace("'", "\""))
        X = torch.tensor([dict_data['data']])
        y_test_hat_softmax = model(X)
        y_test_hat = torch.max(y_test_hat_softmax, 1)
        y_test_class = y_test_hat.indices.cpu().detach().numpy()[0]
        cls_dict = {
            0: 'Iris Setosa',
            1: 'Iris Versicolor',
            2: 'Iris Virginica'
        }
        return f"Predicted class: {cls_dict[y_test_class]}"

if __name__ == '__main__':
    app.run()