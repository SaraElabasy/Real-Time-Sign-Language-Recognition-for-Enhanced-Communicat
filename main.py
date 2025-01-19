from flask import Flask, render_template, request
import os
from skimage.feature import hog
from skimage import io, color, transform
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class EnglishHandler:
    """Handles English-specific functionality."""

    def __init__(self):
        try:
            self.models = {
                'SVM': joblib.load(
                    'svm_asl_model.joblib')  # Update path if necessary
            }
            print("English model loaded successfully.")
        except FileNotFoundError:
            print(
                "Error: Model file 'svm_asl_model.joblib' not found. Please check the path."
            )
            exit(1)
        except Exception as e:
            print(f"Error loading the English model: {e}")
            exit(1)

    def preprocess_and_predict(self, image_path, model):
        """Preprocess the image and predict using the selected model."""
        try:
            # Load the image
            image = io.imread(image_path)

            # Preprocess the image
            image_resized = transform.resize(
                image, (64, 64))  # Resize to match training size
            image_grayscale = color.rgb2gray(
                image_resized)  # Convert to grayscale

            # Extract HOG features
            hog_features = hog(image_grayscale,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               feature_vector=True).reshape(
                                   1, -1)  # Reshape to match model input

            # Predict the label
            predicted_label = model.predict(hog_features)[0]
            return predicted_label
        except Exception as e:
            print(f"Error during preprocessing or prediction (English): {e}")
            return None


class ArabicHandler:
    """Handles Arabic-specific functionality."""

    def __init__(self):
        try:
            self.arabic_class_mapping = {
                0: "ع",
                1: "ا ل",
                2: "ا",
                3: "ب",
                4: "ض",
                5: "د",
                6: "ف",
                7: "غ",
                8: "ح",
                9: "ه",
                10: "ج",
                11: "ك",
                12: "خ",
                13: "لا",
                14: "ل",
                15: "م",
                16: "ن",
                17: "ق",
                18: "ر",
                19: "ص",
                20: "س",
                21: "ش",
                22: "ط",
                23: "ت",
                24: "ة",
                25: "ذ",
                26: "ث",
                27: "و",
                28: "ي",
                29: "ظ",
                30: "ز"
            }
            self.models = {
                'MobileNet_Arabic':
                tf.keras.models.load_model(
                    'mobilenet_arabic_sign_model.h5')  # Load the .h5 model
            }
            print("Arabic MobileNet model loaded successfully.")
        except FileNotFoundError:
            print(
                "Error: Model file 'mobilenet_arabic_sign_language_model.h5' not found. Please check the path."
            )
            exit(1)
        except Exception as e:
            print(f"Error loading the Arabic MobileNet model: {e}")
            exit(1)

    def preprocess_and_predict(self, image_path, model):
        """Preprocess the image and predict using the selected model."""
        try:
            # Load the image
            img = image.load_img(
                image_path,
                target_size=(64,
                             64))  # MobileNet typically uses 224x224 input
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array,
                                       axis=0)  # Add batch dimension
            img_array = preprocess_input(
                img_array)  # Preprocess image for MobileNet

            # Predict the label
            predictions = model.predict(img_array)
            predicted_label = np.argmax(
                predictions)  # Get the class with the highest probability
            predicted_index = np.argmax(predictions)
            predicted_letter = self.arabic_class_mapping[predicted_index]
            return predicted_letter
            # return predicted_label
        except Exception as e:
            print(f"Error during preprocessing or prediction (Arabic): {e}")
            return None

# Initialize handlers
english_handler = EnglishHandler()
arabic_handler = ArabicHandler()


@app.route('/', methods=['GET', 'POST'])
def index_eng():
    """Main route for English uploading and predicting."""
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            return "No file selected. Please upload an image."

        file = request.files['file']

        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Get the selected model
        selected_model_name = request.form.get('model')
        if selected_model_name not in english_handler.models:
            return "Invalid model selected."

        # Load the selected model
        model = english_handler.models[selected_model_name]

        # Predict the label
        predicted_label = english_handler.preprocess_and_predict(
            filepath, model)
        if predicted_label is None:
            return "Error during prediction. Check server logs for details."

        return render_template('result eng.html',
                               label=predicted_label,
                               model=selected_model_name)

    return render_template('index eng.html',
                           models=english_handler.models.keys())


@app.route('/ar', methods=['GET', 'POST'])
def index_ar():
    """Main route for Arabic uploading and predicting."""
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            return "لا يوجد ملف مرفوع! يرجى رفع صورة."

        file = request.files['file']

        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Get the selected model
        selected_model_name = request.form.get('model')
        if selected_model_name not in arabic_handler.models:
            return "النموذج المختار غير صالح."

        # Load the selected model
        model = arabic_handler.models[selected_model_name]

        # Predict the label
        predicted_label = arabic_handler.preprocess_and_predict(
            filepath, model)
        if predicted_label is None:
            return "خطأ أثناء التنبؤ. تحقق من سجلات الخادم للحصول على التفاصيل."

        return render_template('result ar.html',
                               label=predicted_label,
                               model=selected_model_name)

    return render_template('index ar.html',
                           models=arabic_handler.models.keys())

if __name__ == '__main__':
    app.run(debug=True)
