#!/usr/bin/env python

from flask import Flask, jsonify, request, json
import numpy as np
import pickle
import sklearn 
import os
from sklearn.mixture import GaussianMixture 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs


app = Flask(__name__)
@app.route('/home')
def home():
        return "This is the home directory. To upload a model use the curl POST command. To recieve information on a model go to the /models extension of this webpage."


def make_prediction(data):
    filename = "fpath"
    model = None
    with open(filename, 'rb') as istream:
        model = pickle.load(istream)
        print(type(model))
        idata = data['input']
        print(data)
        print(idata)
        input_data = np.array(idata)
        print(input_data)
        # make a prediction
        probabilities = model.predict_proba(input_data)
        output = {'predictions': probabilities}
        return output

def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {convert_key(key): convert_ndarrays(value) for key, value in obj.items()}
    else:
        return obj

def convert_key(key):
    if isinstance(key, (np.integer, np.int64, np.int32)):
        return int(key)
    elif isinstance(key, (np.floating, np.float64, np.float32)):
        return float(key)
    elif isinstance(key, (np.bool_)):
        return bool(key)
    else:
        return key

@app.route('/models', methods = ['GET'])
def obtain_models_info():
        model = request.args.get('model')
        print(model)
        with open("/nfs/chess/id4baux/suchi/XTEC/nathan/CsV3Sb5Snx/ACS5-11-1/CsV3Sb5Snx_ACS5-11-1_CWD_sig0.p" , 'rb') as f:
                pkl_data = pickle.load(f)
                converted_data = convert_ndarrays(pkl_data)
                data = json.dumps(converted_data)
                '''
                records = json.loads(data)
                for record in records:
                    if 'model' not in record:
                        print(f"Record without model key: {record}")  # Debugging statement
                        if record.get('model') == model:
                            #if record['model'] == model:
                            return jsonify(record)
                    return jsonify({'error': 'data not found'})
'''

@app.route('/predict', methods=['POST'])
def predict_post():
        try:
            data = request.get_json()
            print(data)
            prediction = make_prediction(data)
            print(prediction)
            response = {
                   'status': 'success',
                 'prediction': prediction
                 }
            response = convert_ndarrays(response)
            print(response)
            return jsonify(response), 200
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400






@app.route('/upload',methods = ['POST'])
def upload_model():
    # Define the base directory for storing models
    BASE_DIR = 'tmp/nathan-server/models'
    # Function to create directories based on the date and parameters
    def create_directories(base_dir, date, params):
        # Path to the directory for the given date
        date_path = os.path.join(base_dir, 'date')
        # Path to the 'params' directory within the date directory
        params_path = os.path.join(date_path, 'params')        
        # Create the 'params' directory if it doesn't exist
        os.makedirs(params_path, exist_ok=True)        
        # Create subdirectories for each parameter
        for param in params:
            os.makedirs(os.path.join(params_path, param), exist_ok=True)        
        # Return the path to the 'params' directory
        return params_path
    # Check if the request is JSON
    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json"}), 400
    # Get JSON data from the request
    data = request.get_json()
    # Extract required fields from the JSON data
    model_file_name = data.get('model')
    model_name = data.get('model_name')
    upload_time = data.get('time')
    params = data.get('params', [])
    # Validate required fields
    if not model_file_name or not model_name or not upload_time:
        return jsonify({"error": "Missing required fields: model, model_name, and/or time"}), 400
        # Create the necessary directories for the given date and parameters
    params_path = create_directories(BASE_DIR, upload_time, params)
    # Construct the file path to save the model file
    # In a real scenario, you would retrieve the actual file content to save it.
    # Here, we simulate saving the model file with a placeholder content
    with open(model_file_name, 'rb') as f:
        model = f.read 
    model_contents = model
    with open(model_file_path, 'w') as f:
        f.write(model_contents)
 
    # Prepare the metadata for the model
    metadata = {
        "model_name": model_name,
        "time": upload_time,
        "params": params
    }
    # Save the metadata as a JSON file in the same directory as the model file
    metadata_path = os.path.join(params_path, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    # Return a success response
        return jsonify({"message": "Model uploaded successfully"}), 200
    # Return an error response if the upload failed
    return jsonify({"error": "Upload failed"}), 500


def login():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        return json
    else:
        return 'Content-Type not supported!'


if __name__ == '__main__':
        app.run(debug=True, port=2000, host= '0.0.0.0')












'''
    # Get the file from the request
    file = request.files['model']

    # Check if the 'model' part is present in the request
    if 'model' not in request.files:
        return jsonify({"error": "No model file part"}), 400
    

    # Check if a file was actually selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # If a file is selected, proceed with processing
    if file:
        model_file_name = data.get('model')
        # Get the model name from the form data
        model_name = request.form.get('model_name')
        # Get the upload time from the form data
        upload_time = request.form.get('time')
        # Get the parameters from the form data (comma-separated list)
        params = request.form.getlist('params')
        # Split the comma-separated parameters into a list
        params = params[0].split(",") if len(params) > 0 else []
'''



