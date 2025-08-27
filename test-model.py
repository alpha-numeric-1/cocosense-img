import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ==== CONFIGURATION ====
MODEL_PATH = "coconut_model.tflite"
IMAGES_DIR = "images"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["malakanin", "malakatad", "malauhog"] 

# ==== LOAD TFLITE MODEL ====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== FUNCTION TO PREPROCESS IMAGE ====
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0,1] if your model expects it
        # img_array /= 255.0
        
        # Or normalize to [-1,1] if needed:
        # img_array = (img_array / 127.5) - 1
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {str(e)}")

# ==== RUN INFERENCE ====
def classify_image(image_path):
    try:
        img_array = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Apply softmax if output is logits
        # output_data = tf.nn.softmax(output_data).numpy()

        return output_data
    except Exception as e:
        raise RuntimeError(f"Error during inference: {str(e)}")
    
# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    print("Coconut Maturity Classification (Image-based)\n")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Directory '{IMAGES_DIR}' not found!")
        exit(1)

    image_files = [f for f in os.listdir(IMAGES_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the directory.")
        exit(1)

    print("Available images in folder:")
    for idx, filename in enumerate(image_files):
        print(f"  [{idx}] {filename}")

    try:
        choice = int(input("\nEnter the number of the image you want to classify: "))
        if choice < 0 or choice >= len(image_files):
            print("Invalid selection.")
            exit(1)
            
        selected_image = os.path.join(IMAGES_DIR, image_files[choice])
        print(f"\nClassifying: {image_files[choice]}")
        
        # Get predictions using the classify_image function
        predictions = classify_image(selected_image)
        
        if len(predictions) != len(CLASS_NAMES):
            raise ValueError(
                f"Expected {len(CLASS_NAMES)} outputs, got {len(predictions)}"
            )

        # Print results
        print("\nResults:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name}: {predictions[i]:.4f}")
        
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        print(f"  Predicted Class: {CLASS_NAMES[predicted_class_idx]} "
              f"(confidence: {confidence:.4f})")

    except ValueError as ve:
        print(f"Input error: {ve}")
    except Exception as e:
        print(f"Error during classification: {str(e)}")