import os
import datetime
import random
import traceback # Import traceback for debugging
from flask import Flask, request, render_template, jsonify, url_for, session, redirect, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from flask_sqlalchemy import SQLAlchemy
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_12345'

# FIX 1: Use absolute path relative to this script file, not generic CWD
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
db = SQLAlchemy(app)

# --- Directory Setup ---
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
HEATMAP_FOLDER = os.path.join(STATIC_DIR, 'heatmaps')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    age = db.Column(db.String(10), nullable=True)      
    gender = db.Column(db.String(20), nullable=True)
    contact = db.Column(db.String(50), nullable=True) 
    address = db.Column(db.String(200), nullable=True) 

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    hospital_name = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

class ScanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image = db.Column(db.String(200), nullable=False)
    heatmap_image = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text, nullable=False)
    report_date = db.Column(db.DateTime, default=datetime.datetime.now)

def create_db_if_not_exists():
    if not os.path.exists(db_path):
        with app.app_context():
            db.create_all()
            print("Database created.")

# --- LOAD MODELS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Gatekeeper
SCAN_TYPE_MODEL_PATH = os.path.join(BASE_DIR, 'gatekeeper_dataset', 'brain_classifier.pth')
scan_type_model = models.resnet18(weights=None)
scan_type_model.fc = nn.Linear(scan_type_model.fc.in_features, 2)
try:
    # Use weights_only=True or False depending on security, but explicit map_location is safer
    scan_type_model.load_state_dict(torch.load(SCAN_TYPE_MODEL_PATH, map_location=device))
    scan_type_model = scan_type_model.to(device)
    scan_type_model.eval()
    print("Gatekeeper model loaded.")
except Exception as e:
    print(f"WARNING: Gatekeeper model failed to load: {e}")
    scan_type_model = None

# 2. Stroke Model
STROKE_MODEL_PATH = os.path.join(BASE_DIR, 'gatekeeper_dataset', 'all_png_images', 'EfficiencyNet_stroke_classification_6600.pth')
stroke_model = models.resnet18(weights=None) 
stroke_model.fc = nn.Linear(stroke_model.fc.in_features, 3)
try:
    stroke_model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location=device))
    stroke_model = stroke_model.to(device)
    stroke_model.eval()
    print("Stroke model loaded.")
except Exception as e:
    print(f"WARNING: Stroke model failed to load: {e}")

# Define target layer safely
try:
    target_layer = [stroke_model.layer4[-1]]
except AttributeError:
    print("Error: Model structure does not match ResNet (layer4 missing).")
    target_layer = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
stroke_class_names = ['non-stroke', 'hemorrhage', 'ischemic']
scan_type_class_names = ['brain', 'other'] 
explanations = {
    "non-stroke": "No major abnormalities detected. Regular check-ups recommended.",
    "hemorrhage": "Detected irregular pixel intensity patterns consistent with bleeding.",
    "ischemic": "Detected darker regions suggesting reduced blood flow and blockage."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        contact = request.form['contact']
        address = request.form['address']
        
        if User.query.filter_by(username=username).first():
            flash('Username taken', 'danger')
        else:
            new_user = User(username=username, age=age, gender=gender, contact=contact, address=address)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session: return redirect(url_for('login'))
    # Fetch History
    my_scans = ScanRecord.query.filter_by(user_id=session['user_id']).order_by(ScanRecord.report_date.desc()).all()
    my_appointments = Appointment.query.filter_by(user_id=session['user_id']).order_by(Appointment.created_at.desc()).all()
    return render_template('index.html', username=session.get('username'), scans=my_scans, appointments=my_appointments)

@app.route('/settings')
def settings():
    if 'user_id' not in session: return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('settings.html', user=user)

@app.route('/hospitals')
def hospitals():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('hospitals.html')

@app.route('/about')
def about():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/ai_assistant')
def ai_assistant():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('ai_assistant.html')

# --- ADDED THESE MISSING ROUTES ---
@app.route('/help')
def help_page():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('help.html')

@app.route('/features')
def features():
    # Adding this too because features.html links to it
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('features.html')
# ----------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').lower()
    resp = ""
    action = None

    if 'stroke' in msg:
        resp = "A stroke happens when blood flow to the brain is interrupted. Ischemic = Blockage. Hemorrhagic = Bleeding. Immediate action is required."
    elif 'prevent' in msg or 'diet' in msg:
        resp = "To prevent strokes: Control blood pressure, quit smoking, manage diabetes, and maintain a healthy diet low in salt/fat."
    elif 'symptom' in msg:
        resp = "Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency."
    elif 'upload' in msg:
        resp = "I'll take you to the dashboard to upload a scan."
        action = "redirect:/"
    elif 'hospitals' in msg:
        resp = "Opening the hospital locator..."
        action = "redirect:/hospitals"
    elif 'book' in msg or 'appointment' in msg:
        resp = "You can book appointments in the Hospital section. Taking you there..."
        action = "redirect:/hospitals"
    elif 'setting' in msg or 'profile' in msg:
        resp = "Opening settings..."
        action = "redirect:/settings"
    else:
        resp = "I am specialized in stroke assistance. Ask me about symptoms, prevention, or how to use this app."

    return jsonify({'response': resp, 'action': action})

@app.route('/report/<int:scan_id>')
def view_report(scan_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    
    scan = ScanRecord.query.get_or_404(scan_id)
    user = User.query.get(session['user_id'])
    
    if scan.user_id != user.id:
        return "Unauthorized", 403

    rec = "Regular check-ups recommended." if scan.prediction == 'non-stroke' else "Immediate medical attention required."
    
    report_data = {
        'scan_id': scan.id,
        'patient_name': user.username,
        'patient_age': user.age,
        'patient_contact': user.contact,
        'patient_address': user.address,
        'report_date': scan.report_date.strftime("%B %d, %Y - %H:%M"),
        'prediction': scan.prediction,
        'confidence': scan.confidence,
        'description': scan.description,
        'recommendation': rec,
        'original_image_url': url_for('static', filename=f'uploads/{scan.original_image}'),
        'heatmap_url': url_for('static', filename=f'heatmaps/{scan.heatmap_image}')
    }
    return render_template('report.html', data=report_data)

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if 'user_id' not in session: return jsonify({'success': False}), 401
    data = request.get_json()
    try:
        new_appointment = Appointment(
            user_id=session['user_id'],
            hospital_name=data.get('hospital_name'),
            date=data.get('date'),
            time=data.get('time')
        )
        db.session.add(new_appointment)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try: image = Image.open(filepath).convert('RGB')
        except: return jsonify({'error': 'Invalid image.'}), 400

        input_tensor = transform(image).unsqueeze(0).to(device)

        # Gatekeeper Check
        if scan_type_model:
            with torch.no_grad():
                scan_out = scan_type_model(input_tensor)
                probs = torch.softmax(scan_out, dim=1)
                _, scan_pred = torch.max(scan_out, 1)
                
                # Check index bounds to avoid IndexError
                pred_idx = scan_pred.item()
                if pred_idx < len(scan_type_class_names):
                    if scan_type_class_names[pred_idx] == 'other' or probs[0][0].item() < 0.9:
                        return jsonify({'error': 'Non-brain image detected.'}), 400

        # Stroke Analysis
        with torch.no_grad():
            outputs = stroke_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            label = stroke_class_names[pred.item()]
            conf_score = conf.item()

        # FIX 2: Safer Grad-CAM Logic
        if target_layer is not None:
            # Re-initialize GradCAM per request (safe but slightly slow)
            cam = GradCAM(model=stroke_model, target_layers=target_layer)
            
            # Generate mask
            # NOTE: If using a very new version of pytorch-grad-cam, ensure API compatibility
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred.item())])[0, :]
            
            # FIX 3: Convert image to float32 specifically for show_cam_on_image
            vis_img = np.float32(image.resize((224, 224))) / 255.0
            
            # visualization
            visualization = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)
            
            heatmap_filename = f"heatmap_{filename}"
            Image.fromarray(visualization).save(os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename))
        else:
            # Fallback if GradCAM fails to init
            heatmap_filename = filename

        # SAVE SCAN TO DB
        new_scan = ScanRecord(
            user_id=session['user_id'],
            original_image=filename,
            heatmap_image=heatmap_filename,
            prediction=label,
            confidence=f"{conf_score*100:.2f}%",
            description=explanations[label]
        )
        db.session.add(new_scan)
        db.session.commit()

        return jsonify({'success': True, 'redirect_url': url_for('view_report', scan_id=new_scan.id)})

    except Exception as e:
        # FIX 4: Print the full error trace to console so you can debug
        print("!!! ERROR IN /PREDICT !!!")
        traceback.print_exc()
        return jsonify({'error': f'Internal Error: {str(e)}'}), 500

if __name__ == '__main__':
    create_db_if_not_exists()
    # FIX 5: Enable debug mode to see errors in browser
    app.run(host='0.0.0.0', port=8080, debug=True)
