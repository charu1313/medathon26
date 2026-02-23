from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import math
import random

# Serve React App from the 'dist' folder
basedir = os.path.abspath(os.path.dirname(__file__))
dist_folder = os.path.join(basedir, '../frontend/dist')
app = Flask(__name__, static_folder=dist_folder, static_url_path='/')
CORS(app)  # Enable CORS for all routes

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'healthcare.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True) # Nullable for Google Auth users
    role = db.Column(db.String(20), nullable=False) # 'patient' or 'doctor'
    
    # Location
    country = db.Column(db.String(50))
    state = db.Column(db.String(50))
    city = db.Column(db.String(50))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    
    # Doctor Specific
    license_id = db.Column(db.String(50))
    specialty = db.Column(db.String(50))
    is_verified_doctor = db.Column(db.Boolean, default=False)
    
    # Auth & Security
    google_id = db.Column(db.String(100))
    otp_code = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)
    
    is_pregnant = db.Column(db.Boolean, default=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    appointment_date = db.Column(db.String(20), nullable=False) # Store as YYYY-MM-DD
    appointment_time = db.Column(db.String(10), nullable=False) # Store as HH:MM
    status = db.Column(db.String(20), default='pending') # pending, accepted, rejected, completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    patient = db.relationship('User', foreign_keys=[patient_id], backref='appointments_as_patient')
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref='appointments_as_doctor')

# Create tables
with app.app_context():
    db.create_all()


# --- Mock Model for MVP (Replace with real training later) ---
MODEL_PATH = 'maternal_risk_model.pkl'

def train_dummy_model():
    from sklearn.ensemble import RandomForestClassifier
    # Dummy data: [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]
    # RiskLevels: 0=low, 1=mid, 2=high
    X_dummy = np.array([
        [25, 120, 80, 7.0, 98.0, 70],
        [35, 140, 90, 10.0, 98.0, 80],
        [29, 90, 70, 7.5, 100.0, 80],
        [40, 160, 100, 15.0, 98.0, 90],
        [22, 110, 75, 6.8, 98.0, 68]
    ])
    y_dummy = np.array([0, 1, 0, 2, 0]) # 0: Low, 1: Mid, 2: High
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_dummy, y_dummy)
    joblib.dump(clf, MODEL_PATH)
    print("Dummy model trained and saved.")

if not os.path.exists(MODEL_PATH):
    train_dummy_model()

model = joblib.load(MODEL_PATH)


# --- Helper Functions ---
def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula
    R = 6371 # Radius of earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

def verify_license_mock(license_id):
    # Mock verification: simple regex or length check
    return len(license_id) > 5 and license_id.isalnum()


# --- Routes ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    role = data.get('role', 'patient')
    
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User already exists'}), 400

    hashed_password = generate_password_hash(password) if password else None
        
    new_user = User(
        name=name,
        email=email,
        password_hash=hashed_password,
        role=role,
        country=data.get('country'),
        state=data.get('state'),
        city=data.get('city'),
        latitude=float(data.get('latitude', 0)) if data.get('latitude') else None,
        longitude=float(data.get('longitude', 0)) if data.get('longitude') else None,
        is_pregnant=data.get('is_pregnant', False)
    )

    if role == 'doctor':
        license_id = data.get('license_id')
        if not license_id:
             return jsonify({'error': 'License ID required for doctors'}), 400
        
        new_user.license_id = license_id
        new_user.specialty = data.get('specialty')
        
        # Verify license
        if verify_license_mock(license_id):
            new_user.is_verified_doctor = True # Auto-verify for demo if valid format
        else:
            return jsonify({'error': 'Invalid License ID format'}), 400

    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'Registration successful', 'role': role}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    if not user.password_hash or not check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # OTP for Patients
    if user.role == 'patient':
        # Rate limit check: if otp exists and not expired, maybe wait?
        # For now just overwrite
        otp = str(random.randint(100000, 999999))
        user.otp_code = otp
        user.otp_expiry = datetime.utcnow() + timedelta(minutes=5)
        db.session.commit()
        print(f"OTP for {email}: {otp}") # Mock sending SMS
        return jsonify({'message': 'OTP sent', 'require_otp': True, 'email': email, 'debug_otp': otp}), 200

    # Google Auth Only check for Doctors (if configured) - here simple login for verified docs
    if user.role == 'doctor' and not user.is_verified_doctor:
        return jsonify({'error': 'Doctor License not verified'}), 403

    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'role': user.role
        }
    }), 200

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get('email')
    otp = data.get('otp')
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'User not found'}), 400

    if user.otp_code != otp:
        return jsonify({'error': 'Invalid OTP'}), 400
        
    if user.otp_expiry and datetime.utcnow() > user.otp_expiry:
        return jsonify({'error': 'OTP Expired'}), 400
        
    # Clear OTP
    user.otp_code = None
    user.otp_expiry = None
    db.session.commit()
    
    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'role': user.role
        }
    }), 200

@app.route('/api/google-login', methods=['POST'])
def google_login():
    data = request.json
    email = data.get('email')
    google_id = data.get('googleId')
    name = data.get('name')
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        # Auto-register as Doctor via Google? Or reject?
        # Requirement: "Allow doctor to login via Google... Link Google account"
        # Since we need license, we can't fully auto-register without it.
        return jsonify({'error': 'Please register first to verify license'}), 404

    if user.role == 'doctor':
        user.google_id = google_id
        if not user.is_verified_doctor:
             return jsonify({'error': 'License verification pending'}), 403
        
        db.session.commit()
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'role': user.role
            }
        }), 200
        
    return jsonify({'error': 'Google login only for Doctors in this demo'}), 400


@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    city = request.args.get('city')
    
    doctors = User.query.filter_by(role='doctor', is_verified_doctor=True).all()
    results = []
    
    for doc in doctors:
        dist = None
        if lat and lon and doc.latitude and doc.longitude:
            dist = calculate_distance(lat, lon, doc.latitude, doc.longitude)
        
        doc_data = {
            'id': doc.id,
            'name': doc.name,
            'specialty': doc.specialty,
            'city': doc.city,
            'latitude': doc.latitude,
            'longitude': doc.longitude,
            'distance_km': round(dist, 1) if dist is not None else None
        }
        
        if city and doc.city and city.lower() != doc.city.lower():
            continue 
            
        results.append(doc_data)
        
    # Sort by distance
    if lat and lon:
        results.sort(key=lambda x: x['distance_km'] if x['distance_km'] is not None else 9999)
        
    return jsonify(results)


@app.route('/api/appointments', methods=['GET'])
def get_appointments():
    user_id = request.args.get('user_id')
    role = request.args.get('role')
    
    if role == 'doctor':
        appts = Appointment.query.filter_by(doctor_id=user_id).all()
    else:
        appts = Appointment.query.filter_by(patient_id=user_id).all()
        
    results = []
    for a in appts:
        results.append({
            'id': a.id,
            'patient_name': a.patient.name,
            'doctor_name': a.doctor.name,
            'date': a.appointment_date,
            'time': a.appointment_time,
            'status': a.status
        })
        
    return jsonify(results)

@app.route('/api/appointments/book', methods=['POST'])
def book_appointment():
    data = request.json
    new_appt = Appointment(
        patient_id=data['patient_id'],
        doctor_id=data['doctor_id'],
        appointment_date=data['date'],
        appointment_time=data['time']
    )
    db.session.add(new_appt)
    db.session.commit()
    return jsonify({'message': 'Appointment requested'}), 201

@app.route('/api/appointments/<int:id>/status', methods=['PUT'])
def update_appointment(id):
    data = request.json
    status = data.get('status')
    appt = Appointment.query.get(id)
    if appt:
        appt.status = status
        db.session.commit()
        return jsonify({'message': 'Status updated'}), 200
    return jsonify({'error': 'Appointment not found'}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_risk():
    try:
        data = request.json
        features = [
            float(data.get('age', 25)),
            float(data.get('systolic_bp', 120)),
            float(data.get('diastolic_bp', 80)),
            float(data.get('blood_sugar', 7.0)),
            float(data.get('body_temp', 98.0)),
            float(data.get('heart_rate', 70))
        ]
        
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        confidence = max(probabilities) * 100
        
        risk_level = "Low Risk"
        if prediction == 1:
            risk_level = "Medium Risk"
        elif prediction == 2:
            risk_level = "High Risk"
            
        recommendations = generate_recommendations(risk_level, features)
        
        return jsonify({
            'risk_level': risk_level,
            'confidence': f"{confidence:.2f}%",
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate_recommendations(risk, features):
    recs = {
        'diet': [],
        'exercise': [],
        'medical': []
    }
    
    if risk == "High Risk":
        recs['diet'].append("Strictly limit sugar and salt intake.")
        recs['diet'].append("Consult a nutritionist immediately.")
        recs['exercise'].append("Avoid strenuous activity. Bed rest may be advised.")
        recs['medical'].append("Immediate consultation with a specialist is required.")
        recs['medical'].append("Monitor blood pressure every 4 hours.")
    elif risk == "Medium Risk":
        recs['diet'].append("Reduce processed foods and focus on whole grains.")
        recs['diet'].append("Increase hydration.")
        recs['exercise'].append("Light walking (15-20 mins) if approved by doctor.")
        recs['medical'].append("Schedule a follow-up appointment within 3 days.")
    else:
        recs['diet'].append("Maintain a balanced diet rich in leafy greens and protein.")
        recs['exercise'].append("Regular prenatal yoga or walking (30 mins daily).")
        recs['medical'].append("Continue standard prenatal checkups.")
        
    return recs

@app.route('/api/recommendations/diet', methods=['POST'])
def get_diet_plan():
    # Existing logic preserved
    data = request.json
    risk = data.get('risk_level', 'Low Risk')
    plan = {
        'risk_level': risk,
        'calories': '2200-2400 kcal' if risk == 'High Risk' else '2000-2200 kcal',
        'hydration': '3.5 Liters/day' if risk == 'High Risk' else '2.5 Liters/day',
        'allowed': ['Leafy greens (Spinach, Kale)', 'Lean proteins (Chicken, Tofu)', 'Whole grains (Quinoa, Brown Rice)', 'Berries and Citrus fruits'],
        'avoid': ['Processed sugar', 'Excessive salt (>2000mg)', 'Caffeine (>200mg)', 'Raw seafood'],
        'weekly_plan': [
            {'day': 'Monday', 'breakfast': 'Oatmeal with berries', 'lunch': 'Grilled chicken salad', 'dinner': 'Steamed fish with veggies'},
            {'day': 'Wednesday', 'breakfast': 'Greek yogurt & nuts', 'lunch': 'Quinoa bowl', 'dinner': 'Lentil soup'},
            {'day': 'Friday', 'breakfast': 'Scrambled eggs', 'lunch': 'Tuna wrap', 'dinner': 'Grilled tofu & broccoli'}
        ],
        'doctors': [
            {'name': 'Dr. Sarah Smith', 'specialty': 'Nutritionist', 'rating': 4.9, 'location': 'City Hospital', 'link': 'https://www.google.com/search?q=Dr.+Sarah+Smith+Nutritionist'},
            {'name': 'Dr. Emily Roe', 'specialty': 'Dietitian', 'rating': 4.7, 'location': 'Wellness Clinic', 'link': 'https://www.google.com/search?q=Dr.+Emily+Roe+Dietitian'}
        ]
    }
    return jsonify(plan)

@app.route('/api/recommendations/exercise', methods=['POST'])
def get_exercise_plan():
    # Existing logic preserved
    data = request.json
    risk = data.get('risk_level', 'Low Risk')
    videos = [
        {'title': 'Prenatal Yoga for Beginners', 'url': 'https://www.youtube.com/embed/B87FpWdtCfk'},
        {'title': '15 Min Pregnancy Cardio', 'url': 'https://www.youtube.com/embed/ADVpHQaeEok'}
    ]
    if risk == 'High Risk':
        videos = [
            {'title': 'Bed Rest Stretches', 'url': 'https://www.youtube.com/embed/j7E8s4J6_zE'},
            {'title': 'Deep Breathing for Stress', 'url': 'https://www.youtube.com/embed/inpok4MKVLM'}
        ]
    return jsonify({
        'risk_level': risk,
        'safe_exercises': ['Pelvic floor exercises', 'Gentle stretching', 'Breathing techniques'] if risk == 'High Risk' else ['Prenatal Yoga', 'Walking (30 mins)', 'Swimming', 'Light weights'],
        'videos': videos,
        'trainers': [
            {'name': 'FitMom Studio', 'specialty': 'Prenatal Yoga', 'rating': 4.8, 'location': 'Downtown', 'link': 'https://www.google.com/search?q=FitMom+Studio'},
            {'name': 'Yoga with Adriene', 'specialty': 'Online', 'rating': 5.0, 'location': 'YouTube', 'link': 'https://www.youtube.com/user/yogawithadriene'}
        ]
    })

@app.route('/api/recommendations/medical', methods=['POST'])
def get_medical_advice():
    # Existing logic preserved
    data = request.json
    risk = data.get('risk_level', 'Low Risk')
    return jsonify({
        'risk_level': risk,
        'schedule': ['Every 4 weeks (Weeks 4-28)', 'Every 2 weeks (Weeks 28-36)', 'Weekly (Week 36-Birth)'] if risk == 'Low Risk' else ['Every 2 weeks (Weeks 4-28)', 'Weekly (Week 28-Birth)'],
        'symptoms_to_monitor': ['Headache', 'Vision changes', 'Sudden swelling'] if risk == 'High Risk' else ['Mild cramping', 'Fatigue'],
        'emergency_alert': risk == 'High Risk',
        'hospitals': [
            {'name': 'City General Maternity', 'rating': 4.8, 'location': '123 Main St', 'map': 'https://maps.google.com/?q=City+General+Maternity'},
            {'name': 'Sunrise Women\'s Clinic', 'rating': 4.6, 'location': '456 Oak Ave', 'map': 'https://maps.google.com/?q=Sunrise+Womens+Clinic'}
        ]
    })

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
