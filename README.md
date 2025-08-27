# 🖼️ TFLite Image Classification Tester

This project allows you to test a TensorFlow Lite (`.tflite`) classification model on all images inside a folder.  
It will output the predicted class and confidence score for each image.

---

## 🚀 Getting Started

Follow these steps after cloning the repository.

### 1️⃣ Create a Virtual Environment
It's best to work inside a virtual environment to avoid dependency conflicts.

cd to the project "cd F:/cocosense"

python -m venv venv

Activate the Virtual Invironemetn
venv\Scripts\activate

pip install -r requirements.txt

python test_tflite_images.py



├── images/                 # Folder containing test images
├── model.tflite            # Your TFLite model
├── test_tflite_images.py   # Main script
├── requirements.txt        # Python dependencies
└── README.md    