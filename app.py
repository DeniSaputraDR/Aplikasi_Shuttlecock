from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model
from PIL import Image
import base64
import io
from huggingface_hub import hf_hub_download

app = Flask(__name__)


os.environ["HF_HOME"] = "/tmp/huggingface"
TEMP_FOLDER = "/tmp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

model_path = hf_hub_download(
    repo_id="Denisptra/Shuttlecock",
    filename="model_shuttlecock.h5",
    cache_dir="/tmp/huggingface"
)

model = load_model(model_path)

class_names = ['Layak', 'Tidak Layak']

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)

    if prediction[0][0] > 0.5:
        label = 'Tidak Layak'
        confidence = prediction[0][0] * 100
        saran = "Shuttlecock sebaiknya diganti, tidak layak untuk permainan serius."
        visual = "Bulu rusak, bengkok atau tidak merata."
    else:
        label = 'Layak'
        confidence = (1 - prediction[0][0]) * 100
        saran = "Shuttlecock masih dapat digunakan untuk latihan atau pertandingan."
        visual = "Bulu terlihat rapi & utuh."

    return label, confidence, saran, visual


@app.route('/')
def index():
    return render_template('index.html')


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(TEMP_FOLDER, filename)
            file.save(filepath)

            label, confidence, saran, visual = predict_image(filepath)

            image_data = image_to_base64(filepath)

            os.remove(filepath)

            return render_template(
                'index.html',
                image_url=image_data, 
                label=label,
                confidence=confidence,
                saran=saran,
                visual=visual
            )
    return render_template('index.html', error="Upload gagal.")



@app.route('/capture', methods=['POST'])
def capture():
    data_url = request.form['image_data']
    header, encoded = data_url.split(',', 1)
    img_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(img_data))
    filepath = os.path.join(TEMP_FOLDER, "capture.jpg")
    image.save(filepath)
    
    label, confidence, saran, visual = predict_image(filepath)
    
    image_data = image_to_base64(filepath)
    
    os.remove(filepath)
    
    return render_template(
        'index.html',
        image_url=image_data,
        label=label,
        confidence=confidence,
        saran=saran,
        visual=visual
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)