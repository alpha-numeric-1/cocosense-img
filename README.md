# 🖼️ TFLite Image Classification Tester

This project allows you to test a TensorFlow Lite (`.tflite`) classification model on all images inside a folder.  
It will output the predicted class and confidence score for each image.

---

## 🚀 Getting Started

Follow these steps after cloning the repository.

### 1️⃣ Create a Virtual Environment
It's best to work inside a virtual environment to avoid dependency conflicts.

# Navigate to your project directory
cd {project folder path e.g. F:/cocosense}

# Create a virtual environment using Python 3.10
python3.10 -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

### 2️⃣ Install Dependencie
pip install -r requirements.txt

### 3️⃣ Run the Classifier

python test_tflite_images.py



### 📁 Project Structure
├── images/                 # Folder containing test images
├── coconut_model.tflite            # Your TensorFlow Lite model
├── test_model.py   # Main script for image classification
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation



### ✅ Output
For each image in the images/ folder, the script will display:
- Predicted Class
- Confidence Score

Let me know if you'd like to add logging, batch processing, or a GUI layer to this tool. I’d be happy to help you extend it!



├── images/                 # Folder containing test images
├── model.tflite            # Your TFLite model
├── test_tflite_images.py   # Main script
├── requirements.txt        # Python dependencies

└── README.md    

