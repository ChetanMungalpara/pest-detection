import flask
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
import base64
import io
import cv2 # Added for image processing
from PIL import Image # Added for image processing

app = Flask(__name__)

# --- Load Model 2 (Classification) ---
# (This is your existing model)
model_2_classification = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
                'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 
                'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
                'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
                'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# --- Load plant_disease.json ---
with open("plant_disease.json", 'r') as file:
    plant_disease_db = json.load(file)

# --- Load Model 1 (Segmentation) ---
# This dictionary will map your custom function names to a dummy function
# so that the model can load.
#
def dummy_function(y_true, y_pred):
    return 0.0

custom_objects = {
    'dice_loss': dummy_function,
    'iou_score': dummy_function,
    'dice_coef': dummy_function,
    'iou': dummy_function,
    'dice_coefficient': dummy_function  # <-- THIS IS THE FIX
    # Add any other custom function names from your notebook here
}

try:
    # We now pass the custom_objects dictionary to load_model
    model_1_segmentation = tf.keras.models.load_model(
        "models/leaf_segmentation_model.keras",  # Your path
        custom_objects=custom_objects
    )
    
    SEGMENTATION_IMG_SIZE = 256 # !!! IMPORTANT: Change this to match your Model 1's input size
    print("\n*** Segmentation Model 1 loaded successfully. ***\n") # New success message
except Exception as e:
    print("\n--- FATAL ERROR LOADING MODEL 1 ---")
    print(f"Error: {e}")
    print("This error is VERY important. Fix it before continuing.")
    print("!!! Segmentation model not loaded. The app will run without it. !!!\n")
    model_1_segmentation = None

# --- Helper Functions ---

def preprocess_image_for_model_2(image_pil):
    """Pre-processes a PIL image for Model 2 (Classification)."""
    image = image_pil.resize((160, 160)) # Model 2 expects 160x160
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def preprocess_image_for_model_1(image_pil):
    """Pre-processes a PIL image for Model 1 (Segmentation)."""
    # !!! IMPORTANT: Update this to match your Model 1's pre-processing !!!
    image = image_pil.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))
    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0  # Common normalization step
    feature = np.array([image])
    return feature

def decode_base64_image(base64_string):
    """Decodes a base64 image string into a PIL Image."""
    # The string might have a prefix "data:image/jpeg;base64,"
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image_pil = Image.open(io.BytesIO(image_data))
    return image_pil

def encode_image_to_base64(image_array):
    """Encodes a NumPy mask (0s and 1s) into a transparent Data URL."""
    
    # Squeeze and ensure it's a single-channel grayscale mask
    image_array = np.squeeze(image_array)

    # --- THIS IS THE FIX ---
    # Scale the 0/1 mask to 0/255 for the Alpha channel
    # We also need to make sure it's the right type
    alpha_channel = (image_array * 255).astype(np.uint8)
    # --- END OF FIX ---

    # Get dimensions
    h, w = image_array.shape
    
    # Create a new 4-channel (RGBA) image, fully transparent
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Set the color for the diseased spots (Red)
    rgba_image[:, :, 0] = 255 # R
    rgba_image[:, :, 1] = 0   # G
    rgba_image[:, :, 2] = 0   # B
    
    # Set the Alpha channel (transparency)
    rgba_image[:, :, 3] = alpha_channel # This is now 0 or 255
    
    # Convert this RGBA numpy array to a PIL Image
    img_pil = Image.fromarray(rgba_image, 'RGBA')
    
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str


# --- Routes ---

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    """Serves uploaded images (e.g., for the old /upload route, if you keep it)"""
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    """Renders the main home page."""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the new endpoint that home.html's JavaScript calls.
    It handles both upload and webcam scans.
    """
    try:
        data = request.get_json()
        image_base64 = data['image']
        
        # 1. Decode the image
        original_pil_image = decode_base64_image(image_base64)
        
        # --- 2. Run Model 1 (Segmentation) ---
        if model_1_segmentation:
            # Pre-process for Model 1
            input_tensor_m1 = preprocess_image_for_model_1(original_pil_image)
            
            # Predict mask
            mask_pred = model_1_segmentation.predict(input_tensor_m1)[0]
            
            # Post-process mask (e.g., apply threshold)
            # This assumes your model outputs values between 0 and 1
            mask_thresholded = (mask_pred > 0.5).astype(np.uint8) 
            
            # Calculate damage percentage
            # This is the percentage of *non-zero* pixels in the mask
            percentage = (np.sum(mask_thresholded) / mask_thresholded.size) * 100
            damage_percentage = round(percentage, 2)
            
            # Encode mask to base64 to send to frontend
            # We resize the mask to match the original image's aspect ratio for a good overlay
            mask_resized = cv2.resize(mask_thresholded, original_pil_image.size, interpolation=cv2.INTER_NEAREST)
            mask_data_url = encode_image_to_base64(mask_resized)
        else:
            # Fallback if Model 1 failed to load
            damage_percentage = 0.0
            mask_data_url = "" # No mask
            print("Skipping segmentation prediction.")

        # --- 3. Run Model 2 (Classification) ---
        # Pre-process for Model 2
        input_tensor_m2 = preprocess_image_for_model_2(original_pil_image)
        
        # Predict class
        prediction_array = model_2_classification.predict(input_tensor_m2)
        predicted_class_index = prediction_array.argmax()
        predicted_class_name = class_labels[predicted_class_index]

        # Get cure info
        disease_info = plant_disease_db[predicted_class_index]
        # Format plant and disease names
        if "___" in predicted_class_name:
            plant_name, disease_name = predicted_class_name.split("___")
            plant_name = plant_name.replace("_", " ")
            disease_name = disease_name.replace("_", " ")
        else:
            plant_name = predicted_class_name.replace("_", " ")
            disease_name = "Healthy" # Or "N/A"

        final_cure = disease_info['cure']
        
        # Check if Model 2 detected a background image
        if predicted_class_name == 'Background_without_leaves':
            damage_percentage = 0.0 
            mask_data_url = ""       
            plant_name = "N/A"
            disease_name = "No Leaf Detected"
            final_cure = "Please upload an image of a plant leaf."
        # --- END OF NEW FILTER ---

        # --- 4. Send JSON Response ---
        return jsonify({
            "success": True,
            "original_image": image_base64, # Echo back the original image
            "mask_image": mask_data_url,     # The new mask
            "damage_percentage": damage_percentage,
            "plant_name": plant_name,
            "disease_name": disease_name,
            "cure": final_cure # Use the new cure variable
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)