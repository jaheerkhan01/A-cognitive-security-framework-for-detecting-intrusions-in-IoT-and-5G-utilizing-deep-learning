from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import csv
import os
from io import BytesIO
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)



# Assuming this function is defined within your Flask application
def process_file(filepath):
    instance_df = pd.read_csv(filepath)
    scaler = load('scaler.joblib')
    scaled_instance = scaler.transform(instance_df)
    return scaled_instance[0]

def normalize_features(features):
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    normalized = (features - min_vals) / (max_vals - min_vals)
    return normalized

def feature_to_rgb(value):
    # Assuming value is normalized [0, 1], scale it to 24-bit color
    rgb_value = int(value * 16777215)  # 2^24 - 1
    return (rgb_value >> 16) & 0xFF, (rgb_value >> 8) & 0xFF, rgb_value & 0xFF

def distribute_features_to_rgb_image(normalized_features, image_height=64, image_width=64):
    # Only used for RGB mode
    total_pixels = image_height * image_width
    pixels_per_feature = total_pixels // len(normalized_features)
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    pixel_index = 0
    for feature_value in normalized_features:
        rgb = feature_to_rgb(feature_value)
        for _ in range(pixels_per_feature):
            x = pixel_index // image_width
            y = pixel_index % image_width
            if x < image_height:
                image[x, y] = rgb
            pixel_index += 1

    # Handle any remaining pixels
    if pixel_index < total_pixels:
        remaining_rgb = feature_to_rgb(normalized_features[-1])
        while pixel_index < total_pixels:
            x = pixel_index // image_width
            y = pixel_index % image_width
            image[x, y] = remaining_rgb
            pixel_index += 1

    return image

def features_to_rgb_image(features, image_height=64, image_width=64):
    normalized_features = normalize_features(features)
    return distribute_features_to_rgb_image(normalized_features, image_height, image_width)

def expand_features_to_grayscale_image(features, image_height=64, image_width=64):
    # Normalize the features to range [0, 1]
    normalized_features = normalize_features(features)

    # Calculate the number of features to be mapped per dimension
    features_per_dim = int(np.ceil(np.sqrt(len(normalized_features))))

    # Create a grid for the original feature dimensions
    x = np.linspace(0, features_per_dim - 1, features_per_dim)
    y = np.linspace(0, features_per_dim - 1, features_per_dim)
    z = np.zeros((features_per_dim, features_per_dim))

    # Assign feature values to the grid
    feature_index = 0
    for i in range(features_per_dim):
        for j in range(features_per_dim):
            if feature_index < len(normalized_features):
                z[i, j] = normalized_features[feature_index]
                feature_index += 1

    # Interpolate to fit the image dimensions
    interp_func = interp2d(x, y, z, kind='cubic')
    x_new = np.linspace(0, features_per_dim - 1, image_width)
    y_new = np.linspace(0, features_per_dim - 1, image_height)
    z_new = interp_func(x_new, y_new)

    return z_new

def expand_features_to_grayscale_image2(features, image_height=64, image_width=64):
    normalized_features = normalize_features(features)
    features_per_dim = int(np.ceil(np.sqrt(len(normalized_features))))
    x = np.linspace(0, features_per_dim - 1, features_per_dim)
    y = np.linspace(0, features_per_dim - 1, features_per_dim)

    # Create a meshgrid and flatten it for RegularGridInterpolator
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel()]).T
    values = np.zeros(X.shape).ravel()

    # Fill in the feature values
    for i, val in enumerate(normalized_features):
        if i < len(values):
            values[i] = val

    # Create the interpolator object
    my_interpolator = RegularGridInterpolator((x, y), values.reshape(X.shape), method='linear', bounds_error=False, fill_value=None)

    # New grid
    x_new = np.linspace(0, features_per_dim - 1, image_width)
    y_new = np.linspace(0, features_per_dim - 1, image_height)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')

    # Interpolate
    Z_new = my_interpolator((X_new.ravel(), Y_new.ravel())).reshape(image_height, image_width)

    return Z_new

model_rgb_path = 'models\model_rgb (1).h5' 
model_gray_path = 'models\model_gray (1).h5' 
model_rgb = load_model(model_rgb_path)
model_gray = load_model(model_gray_path)

def preprocess_image(image_file, grayscale=False):
    """
    Preprocesses the image file to the format required by the models.

    Parameters:
    - image_file: The image file to process.
    - grayscale: A boolean indicating if the image is grayscale.

    Returns:
    - A preprocessed image array.
    """
    if grayscale:
        # Load the image, ensuring it's in grayscale
          img = image.load_img(image_file, color_mode='grayscale', target_size=(64, 64))
          img_array = image.img_to_array(img)
          # If grayscale, ensure there's an extra dimension to simulate the channel
          img_array = np.expand_dims(img_array, axis=-1)
    else:
        # Load the image, ensuring it's in RGB
        img = image.load_img(image_file, color_mode='rgb', target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0
    return img_array

def ensemble_predictions(rgb_pred, gray_pred, rgb_weight=0.98, gray_weight=0.94):
    # Normalize the weights so they sum to 1
    total_weight = rgb_weight + gray_weight
    normalized_rgb_weight = rgb_weight / total_weight
    normalized_gray_weight = gray_weight / total_weight
    
    # Calculate weighted average of predictions
    combined_pred = (rgb_pred * normalized_rgb_weight) + (gray_pred * normalized_gray_weight)
    
    # Assuming binary classification with sigmoid activation, threshold at 0.5 to decide the class
    combined_class = (combined_pred > 0.54).astype(int)
    return combined_class

output_dir2='output'
output_dir = 'static/' + 'images'
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'inputFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['inputFile']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file to disk
        upload_path = os.path.join(output_dir, 'uploaded_file.csv')
        file.save(upload_path)
        
        # Process the uploaded file
        scaled_instance = process_file(upload_path)
        
        features = []
        with open(upload_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                values = [float(x) for x in row]
                features.extend(values)

        # Save the generated grayscale image
        image_path = os.path.join(output_dir, 'generated_gray.png')
        gray_img = expand_features_to_grayscale_image2(features)
        plt.imshow(gray_img, cmap='gray')
        plt.axis('off')
        plt.savefig(image_path)
        plt.close()

        # Save the generated RGB image
        image_path2 = os.path.join(output_dir, 'generated_rgb.png')
        rgb = features_to_rgb_image(features)
        plt.imshow(rgb)
        plt.axis('off')
        plt.savefig(image_path2)
        plt.close()

        rgb_file = 'static/images/generated_rgb.png'
        gray_file = 'static/images/generated_gray.png'
        rgb_img = preprocess_image(rgb_file, grayscale=False)
        gray_img = preprocess_image(gray_file, grayscale=True)
        gray_img1 = rgb_img
        # Make predictions
        rgb_pred = model_rgb.predict(rgb_img)
        gray_pred = model_gray.predict(gray_img1)

        # Combine predictions (ensemble)
        combined_pred = ensemble_predictions(rgb_pred, gray_pred)
        prediction_file = os.path.join(output_dir2, 'prediction.txt')
        with open(prediction_file, 'w') as f:
            f.write(str(combined_pred))

        with open(prediction_file, 'r') as f:
            prediction_text = f.read()
        
        # Determine the message based on the prediction
        if prediction_text.strip() == '[[0]]':
            message = 'BENIGN'
        else:
            message = 'ATTACK'
        return render_template('index.html', image=output_dir + '/generated_rgb.png',prediction=message)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/download')
def download_image():
    image_path = os.path.join(output_dir, 'generated_gray.png')
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'})
    return send_file(image_path, as_attachment=True)

if __name__== '__main__':
    app.run(debug=True)