import torch
from flask import Flask, request, jsonify
import ML
from TFRS import net, model_path, device, test_transform, labels

if __name__ == '__main__':
    # 预测
    app = Flask(__name__)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)


    @app.route('/predict', methods=['POST'])
    def predict():
        image = request.files['image']
        predicted_class = ML.image_classification_predict(net, labels, image, test_transform, device)
        response = jsonify({'predicted_class': predicted_class})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


    app.run()
