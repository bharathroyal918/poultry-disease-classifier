# ---------- app.py ----------
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Folders
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('healthy_vs_rotten.h5')
disease_labels = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message="No selected file")
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224,224))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)[0]
            pred_idx = np.argmax(preds)
            pred_label = disease_labels[pred_idx]
            conf_score = preds[pred_idx] * 100

            # Basic treatment recommendations
            recommendations = {
                'Healthy': "No disease detected. Maintain good hygiene, proper ventilation, and nutrition.",
                'Coccidiosis': "Treat with anticoccidial drugs, improve litter management, and provide clean water.",
                'Salmonella': "Isolate affected birds, maintain cleanliness, and consult a vet. Consider antibiotics (per vet recommendation).",
                'NewCastle': "Isolate sick birds, vaccinate healthy ones, disinfect area, and alert local authorities."
            }
            suggestion = recommendations.get(pred_label, "No recommendation found.")

            return render_template(
                'result.html',
                prediction=pred_label,
                score=f"{conf_score:.2f}",
                img_path=filepath,
                suggestion=suggestion
            )
        else:
            return render_template('index.html', message="Invalid file extension.")

    return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)