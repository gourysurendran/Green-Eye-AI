from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
# v1.0.1 - Binary Sanitizer Upgrade
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Suppress TensorFlow C++ Info/Warning messages before TF is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils
import logging
import warnings
from sqlalchemy.exc import LegacyAPIWarning

# Silence LegacyAPIWarning from SQLAlchemy
warnings.filterwarnings("ignore", category=LegacyAPIWarning)

# Configure Logging IMMEDIATELY
import json
import random
from disease_data import disease_info
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configure Logging IMMEDIATELY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

app = Flask(__name__)
# Use an environment variable for the secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'green_eye_fallback_dev_key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///green_eye.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db.init_app(app) called lower after logging...

# ---------------- DATABASE MODELS ----------------

class Farmer(db.Model):
    __tablename__ = 'farmers'
    id = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    password = db.Column(db.String(100), nullable=False)
    language = db.Column(db.String(10), default='en')
    crops = db.Column(db.String(500))
    
    # Relationships
    alert_settings = db.relationship('AlertSettings', backref='farmer', uselist=False, cascade="all, delete-orphan")
    daily_logs = db.relationship('DailyLog', backref='farmer', lazy='dynamic', cascade="all, delete-orphan")

class AlertSettings(db.Model):
    __tablename__ = 'alert_settings'
    farmer_id = db.Column(db.String(20), db.ForeignKey('farmers.id'), primary_key=True)
    crop = db.Column(db.String(100))
    state = db.Column(db.String(100))
    district = db.Column(db.String(100))
    soil = db.Column(db.String(100))
    growth_stage = db.Column(db.String(100))
    sowing_date = db.Column(db.String(50))
    farming_style = db.Column(db.String(100))
    land = db.Column(db.String(100))
    method = db.Column(db.String(100))
    problems = db.Column(db.String(100))
    water = db.Column(db.String(100))
    weather_sync = db.Column(db.String(10), default='No')
    acres = db.Column(db.String(20))

class DailyLog(db.Model):
    __tablename__ = 'daily_logs'
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.String(20), db.ForeignKey('farmers.id'), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    entry = db.Column(db.Text, nullable=False)

class PriceAlert(db.Model):
    __tablename__ = 'price_alerts'
    id = db.Column(db.String(50), primary_key=True)
    farmer_id = db.Column(db.String(20), db.ForeignKey('farmers.id'), nullable=False)
    commodity = db.Column(db.String(100))
    state = db.Column(db.String(100))
    target_price = db.Column(db.Integer)
    created_at = db.Column(db.String(50))
    
    # Relationship with Farmer
    farmer_rel = db.relationship('Farmer', backref=db.backref('price_alerts_list', cascade="all, delete-orphan"))

db.init_app(app)
with app.app_context():
    logging.info(f"Using Database at: {app.config['SQLALCHEMY_DATABASE_URI']}")
    db.create_all()
    
    # Simple migration from farmers.json
    if os.path.exists("farmers.json"):
        try:
            with open("farmers.json", "r") as f:
                farmers_data = json.load(f)
                for fid, fdata in farmers_data.items():
                    if not Farmer.query.get(fid):
                        new_f = Farmer(
                            id=fid,
                            name=fdata.get('name', 'Unknown'),
                            phone=fdata.get('phone', ''),
                            password=fdata.get('password', 'password'),
                            language=fdata.get('language', 'en'),
                            crops=fdata.get('crops', '')
                        )
                        db.session.add(new_f)
                        
                        # Migrate alert settings if they exist
                        if 'alert_settings' in fdata:
                            aset = fdata['alert_settings']
                            new_aset = AlertSettings(
                                farmer_id=fid,
                                crop=aset.get('crop'),
                                state=aset.get('state'),
                                district=aset.get('district'),
                                soil=aset.get('soil'),
                                growth_stage=aset.get('growth_stage'),
                                sowing_date=aset.get('sowing_date'),
                                farming_style=aset.get('farming_style'),
                                land=aset.get('land'),
                                method=aset.get('method'),
                                problems=aset.get('problems'),
                                water=aset.get('water'),
                                weather_sync=aset.get('weather_sync', 'No'),
                                acres=str(aset.get('acres', '1'))
                            )
                            db.session.add(new_aset)
                db.session.commit()
                logging.info("Migrated data from farmers.json to database.")
        except Exception as e:
            logging.warning(f"Auto-migration failed: {e}")
    
    logging.info("Database initialized successfully.")

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "plant_disease_model.h5"
BINARY_MODEL_PATH = "binary_leaf_model.h5"

# ---------------- LOAD MODELS ----------------
model = None
binary_model = None
sanitizer_model = None
class_names = []

try:
    # Class names
    class_names = [
        'Pepper__bell___Bacterial_spot', 
        'Pepper__bell___healthy',
        'Potato___Early_blight', 
        'Potato___Late_blight', 
        'Potato___healthy',
        'Tomato___Bacterial_spot', 
        'Tomato___Early_blight', 
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 
        'Tomato___Spider_mites',
        'Tomato___Target_Spot', 
        'Tomato___Tomato_mosaic_virus',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___healthy',
        'Unknown_Plant_Disease'
    ]

    # --- CUSTOM OBJECTS FOR COMPATIBILITY ---
    def custom_input_layer(**kwargs):
        # Filter out 'batch_shape' which causes issues in some Keras versions
        if 'batch_shape' in kwargs:
            if 'input_shape' not in kwargs:
                # [None, 224, 224, 3] -> (224, 224, 3)
                kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
            else:
                kwargs.pop('batch_shape')
        return tf.keras.layers.InputLayer(**kwargs)

    class DTypePolicy:
        def __init__(self, name='float32', *args, **kwargs): 
            self.name = name
            self.compute_dtype = name
            self.variable_dtype = name
            
        @classmethod
        def from_config(cls, config): 
            return cls(**config)

    custom_objects = {'InputLayer': custom_input_layer, 'DTypePolicy': DTypePolicy}
    
    # Load Main Model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            logging.info(f"Main Model loaded from {MODEL_PATH}")
        except Exception as e:
            logging.warning(f"Standard load failed, trying without compile: {e}")
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            logging.info(f"Main Model loaded (no compile) from {MODEL_PATH}")
        
        # Class names
        class_names = [
            'Pepper__bell___Bacterial_spot', 
            'Pepper__bell___healthy',
            'Potato___Early_blight', 
            'Potato___Late_blight', 
            'Potato___healthy',
            'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites',
            'Tomato___Target_Spot', 
            'Tomato___Tomato_mosaic_virus',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
            'Tomato___healthy',
            'Unknown_Plant_Disease'
        ]
    
    # Load Binary Model
    if os.path.exists(BINARY_MODEL_PATH):
        try:
            binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH, custom_objects=custom_objects)
            logging.info(f"Binary Sanitizer Model loaded from {BINARY_MODEL_PATH}")
        except Exception as e:
            binary_model = tf.keras.models.load_model(BINARY_MODEL_PATH, custom_objects=custom_objects, compile=False)
            logging.info(f"Binary Sanitizer Model loaded (no compile) from {BINARY_MODEL_PATH}")

    # Load MobileNetV2 for generic leaf detection (validation)
    # Using 'imagenet' weights to detect if the object is even a plant/leaf
    sanitizer_model = MobileNetV2(weights='imagenet', include_top=True)
    logging.info("MobileNetV2 sanitizer loaded successfully")

except Exception as e:
    logging.error(f"Error loading models: {e}")

# ---------------- CONTEXT PROCESSORS ----------------
@app.context_processor
def inject_user():
    return dict(
        current_farmer_id=session.get('farmer_id'),
        current_farmer_name=session.get('farmer_name')
    )

# ---------------- ROUTES ----------------

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/dashboard")
def dashboard():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    farmer_id = session.get('farmer_id')
    farmer = Farmer.query.get(farmer_id)
    if not farmer:
        return redirect(url_for('logout'))

    alert_settings = None
    if farmer.alert_settings:
        # Convert SQLAlchemy object to dict for consistency with existing template
        alert_settings = {c.name: getattr(farmer.alert_settings, c.name) for c in farmer.alert_settings.__table__.columns}
    
    success_msg = session.pop('post_register_success', None)
    
    return render_template("dashboard.html", 
                           farmer=farmer.name, 
                           alert_settings=alert_settings,
                           success=success_msg)

@app.route("/irrigation")
def irrigation():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    return render_template("irrigation.html", farmer=session.get('farmer_name'))

@app.route("/profile")
def profile():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    farmer_id = session.get('farmer_id')
    farmer = Farmer.query.get(farmer_id)
    
    if not farmer:
        return redirect(url_for('logout'))
        
    return render_template("profile.html", farmer=farmer, farmer_id=farmer_id)

@app.route("/update-profile", methods=["POST"])
def update_profile():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    farmer_id = session.get('farmer_id')
    farmer = Farmer.query.get(farmer_id)
    if farmer:
        farmer.crops = request.form.get('crops', '')
        db.session.commit()
    
    return redirect(url_for('profile'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.form
        login_input = data.get("farmer_id", "").strip()
        password = data.get("password")
        
        # Try finding by ID first
        farmer = Farmer.query.get(login_input.upper())
        
        # If not found, try finding by phone
        if not farmer:
            farmer = Farmer.query.filter_by(phone=login_input).first()
            
        if farmer and farmer.password == password:
            session['farmer_id'] = farmer.id
            session['farmer_name'] = farmer.name
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Invalid Farmer ID/Phone or Password.")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.form
        name = data.get("name")
        phone = data.get("phone")
        password = data.get("password")
        language = data.get("language", "en")
        crops = data.get("crops", "")
        
        # Generate a simple Farmer ID: FE + last 4 of phone + random
        import random
        phone_digits = "".join([c for c in phone if c.isdigit()])
        phone_suffix = phone_digits[-4:] if len(phone_digits) >= 4 else phone_digits
        farmer_id = f"FE{phone_suffix}{random.randint(10, 99)}".upper()
        
        if Farmer.query.get(farmer_id):
            farmer_id += str(random.randint(1, 9)) # Tiny collision check
            
        new_farmer = Farmer(
            id=farmer_id,
            name=name,
            phone=phone,
            password=password,
            language=language,
            crops=crops
        )
        db.session.add(new_farmer)
        db.session.commit()
        
        # Auto-login after registration
        session['farmer_id'] = farmer_id
        session['farmer_name'] = name
        session['post_register_success'] = f"Welcome {name}! Your Farmer ID is {farmer_id}. Please note it down."
        
        return redirect(url_for('dashboard'))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('welcome'))

# --- LEAF DISEASE DETECTION ---
@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        logging.info("Received prediction request") 
        if "file" not in request.files:
            logging.warning("No file in request") 
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            logging.warning("Empty filename") 
            return jsonify({"error": "No file selected"}), 400

        try:
            if model is None:
                logging.error("Model is None") 
                return jsonify({"error": "Model not loaded"}), 500

            logging.info(f"Processing image: {file.filename}") 
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))
            img_array_raw = np.array(img).astype(np.float32) # 0-255
            
            # 1. LEAF SANITY CHECK - Is it a leaf?
            if sanitizer_model:
                img_sanitizer = preprocess_input(img_array_raw.copy())
                s_preds = sanitizer_model.predict(np.expand_dims(img_sanitizer, axis=0))
                decoded_preds = decode_predictions(s_preds, top=5)[0]
                
                # Keywords that suggest it's a plant, fruit, or vegetable
                plant_keywords = [
                    'leaf', 'plant', 'tree', 'flower', 'fruit', 'vegetable', 'nature', 
                    'grass', 'corn', 'mushroom', 'fungus', 'crocus', 'daisy', 'rose',
                    'pot', 'potted', 'buckeye', 'cabbage', 'broccoli', 'zucchini', 
                    'cucumber', 'bell pepper', 'pomegranate', 'pineapple', 'banana',
                    'lemon', 'orange', 'strawberry', 'apple', 'fig', 'hay', 'acorn',
                    'leafhopper', 'cardoon', 'artichoke', 'earthstar', 'hen-of-the-woods',
                    'lettuce', 'spinach', 'kale', 'orchid', 'sunflower', 'viola', 'clover',
                    'wheat', 'rapeseed', 'lavender', 'mint', 'basil', 'oregano', 'thyme',
                    'parsley', 'cilantro', 'rosemary', 'sage', 'dill'
                ]
                
                is_plant = False
                top_label = ""
                for _, label, score in decoded_preds:
                    label_clean = label.lower().replace('_', ' ')
                    if any(kw in label_clean for kw in plant_keywords):
                        is_plant = True
                        top_label = label_clean
                        break
                    # If it's a very clear common non-plant object, we are more suspicious
                    if score > 0.4 and any(bad_kw in label_clean for bad_kw in ['suit', 'person', 'human', 'face', 'shirt', 'jersey', 'cellular telephone']):
                        is_plant = False
                        break
                
                logging.info(f"Sanitizer Predictions: {decoded_preds}")
                if not is_plant:
                    logging.warning(f"Image rejected by sanitizer. Top pred: {decoded_preds[0]}")
                    return jsonify({
                        "error": "Not a leaf", 
                        "message": "We couldn't detect a plant leaf in your photo. Please upload a clear photo of a single leaf."
                    }), 400

            img_array_norm = img_array_raw / 255.0          # 0-1
            
            # Predict with Main Model (expects 0-1)
            logging.info("Running main model prediction...") 
            preds = model.predict(np.expand_dims(img_array_norm, axis=0))[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx]) * 100
            predicted_class = class_names[idx] if idx < len(class_names) else f"Unknown_Index_{idx}"

            # Binary Sanity Check
            binary_is_healthy = False
            binary_pred = 0.5 
            if binary_model:
                # Binary model has a Rescaling layer, so it NEEDS raw 0-255 input
                binary_pred = float(binary_model.predict(np.expand_dims(img_array_raw, axis=0))[0][0])
                binary_is_healthy = binary_pred > 0.4 
                logging.info(f"Binary Check: {'Healthy' if binary_is_healthy else 'Diseased'} (Score: {binary_pred:.4f})")

            logging.info(f"Multi-Class Prediction: {predicted_class}, Confidence: {confidence:.2f}%") 
            
            # Formulate response
            mc_says_healthy = "healthy" in predicted_class.lower()
            # If multi-class is very sure it's healthy, we trust it more
            mc_is_very_sure_healthy = mc_says_healthy and confidence > 85
            
            mc_is_unknown = "unknown" in predicted_class.lower() or confidence < 35
            
            # Binary checks (Inverted logic: binary_pred is score for 'healthy')
            binary_is_sure_diseased = (binary_model is not None) and (binary_pred < 0.2)
            binary_is_sure_healthy = (binary_model is not None) and (binary_pred > 0.8)

            # DECISION LOGIC - ENHANCED ACCURACY
            if mc_is_very_sure_healthy:
                status = "Healthy"
                predicted_class = "General_Healthy_Plant"
            elif binary_is_sure_diseased and not mc_says_healthy:
                # If binary is sure it's diseased and MC doesn't say healthy, it's diseased
                status = "Diseased"
            elif binary_is_sure_healthy:
                # If binary is sure it's healthy, and MC isn't extremely sure about a specific disease
                if not (not mc_says_healthy and confidence > 90):
                    status = "Healthy"
                    predicted_class = "General_Healthy_Plant"
                else:
                    status = "Diseased"
            elif mc_says_healthy:
                # Moderate healthy confidence
                status = "Healthy"
                predicted_class = "General_Healthy_Plant"
            elif mc_is_unknown:
                # Fallback to binary if MC is unsure
                if binary_pred < 0.4:
                    status = "Diseased"
                    predicted_class = "Unknown_Plant_Disease" # Still diseased but unknown type
                else:
                    status = "Healthy"
                    predicted_class = "General_Healthy_Plant"
            else:
                # Default to Diseased if MC thinks so and binary doesn't strongly object
                status = "Diseased"
            
            # Get detailed info
            details = disease_info.get(predicted_class, {
                'symptoms': 'Vibrant green leaves, strong stem.' if status == "Healthy" else 'No specific info available.',
                'cause': 'N/A' if status == "Healthy" else 'Unknown',
                'treatment': 'Continue regular monitoring.' if status == "Healthy" else 'Consult an expert.',
                'fertilizer': 'Standard Organic Compost.' if status == "Healthy" else 'N/A',
                'retrievable': True
            })

            # Force affected_pct to 0 for healthy plants
            if status == "Healthy":
                affected_pct = 0
            else:
                # Deterministic Severity Calculation
                # Usage: Same image -> Same severity score.
                
                # Create a local random generator seeded by the image content sum
                # This ensures restarting the server or rescanning gives consistent results for the same image.
                seed_val = int(np.sum(img_array_raw)) 
                rng = random.Random(seed_val)
                
                # Base random severity (40 - 75)
                base_random = rng.uniform(40, 75)
                
                # Confidence bonus
                confidence_bonus = 0
                if confidence > 85:
                    confidence_bonus = rng.uniform(5, 12)
                
                # Binary score check
                binary_bonus = 0
                if binary_model and binary_pred < 0.1:
                    binary_bonus = rng.uniform(5, 10)
                
                raw_affected = base_random + confidence_bonus + binary_bonus
                affected_pct = min(96, max(38, raw_affected))

            # Determine final retrievability
            # 1. Start with the database default for that disease
            is_retrievable = details.get('retrievable', True)
            
            # 2. Logic: If the physical damage (affected_pct) is too high (> 85%), 
            # it's usually too late to save that specific leaf/plant area.
            if affected_pct > 85:
                is_retrievable = False

            result = {
                "disease": predicted_class.replace("__", " ").replace("_", " "),
                "status": status,
                "confidence": int(round(confidence)), 
                "symptoms": details['symptoms'],
                "cause": details['cause'],
                "treatment": details['treatment'],
                "fertilizer": details.get('fertilizer', 'N/A'),
                "retrievable": is_retrievable,
                "affected_pct": int(round(affected_pct)) 
            }
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}") 
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return render_template("disease.html")

# --- CROP RECOMMENDATION ---
@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    if request.method == "POST":
        data = request.json
        state = data.get("state")
        district = data.get("district")
        soil_type = data.get("soil_type") 
        season = data.get("season") # NEW
        water = data.get("water")   # NEW
        land = data.get("land")     # NEW
        
        recommendations = utils.get_crop_recommendations(state, district, soil_type, season, water, land)
        return jsonify({"crops": recommendations})

    # Pass locations to template for dropdowns
    locations = utils.get_states_and_districts()
    return render_template("crop_recommendation.html", locations=locations)

# --- FERTILIZER GUIDANCE ---
@app.route("/api/typical-values", methods=["POST"])
def typical_values():
    data = request.json
    soil = data.get("soil_type")
    crop = data.get("crop_type")
    values = utils.get_typical_soil_values(soil, crop)
    if values:
        return jsonify(values)
    return jsonify({"error": "No data available"}), 404

@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    if request.method == "POST":
        data = request.json
        try:
            n = float(data.get("n"))
            p = float(data.get("p"))
            k = float(data.get("k"))
            temp = float(data.get("temperature"))
            humidity = float(data.get("humidity"))
            moisture = float(data.get("moisture"))
            soil = data.get("soil_type")
            crop = data.get("crop_type")
            
            # Get comprehensive recommendation
            result = utils.get_fertilizer_recommendation(
                n, p, k, temp, humidity, moisture, soil, crop
            )
            return jsonify(result)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid or missing input values"}), 400

    return render_template("fertilizer.html")

# --- OTHER PAGES ---
@app.route("/crop-calendar")
def crop_calendar():
    return render_template("crop_calendar.html")

@app.route("/monthly-calendar")
def monthly_calendar():
    return render_template("monthly_calendar.html")

@app.route("/financial")
def financial():
    return render_template("financial.html")

@app.route("/market-insights")
def market_insights():
    locations = utils.get_states_and_districts()
    return render_template("market_insights.html", locations=locations)

@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    farmer_id = session['farmer_id']
    farmer = Farmer.query.get(farmer_id)
    
    if not farmer:
         return redirect(url_for('login'))
    
    locations = utils.get_states_and_districts()
    
    if request.method == "POST":
        data = request.json
        action = data.get('action')
        
        # Scenario 1: Reset Settings (User wants to change crop)
        if action == 'reset_settings':
            if farmer.alert_settings:
                # We can either clear values or delete. 
                # Keeping the record but setting crop to None is often better, or delete:
                db.session.delete(farmer.alert_settings)
                db.session.commit()
            return jsonify({"status": "success"})
            
        # Scenario 2: Save Daily Answer
        elif action == 'save_daily':
            answer = data.get('answer')
            date_key = datetime.now().strftime('%Y-%m-%d')
            
            # Use ORM to update or create
            log_entry = DailyLog.query.filter_by(farmer_id=farmer_id, date=date_key).first()
            if log_entry:
                log_entry.entry = answer
            else:
                new_log = DailyLog(farmer_id=farmer_id, date=date_key, entry=answer)
                db.session.add(new_log)
            db.session.commit()
            return jsonify({"status": "success"})
        
        # Scenario 2b: Get Daily Logs (for Updates Dashboard)
        elif action == 'get_daily_logs':
            logs_dict = {log.date: log.entry for log in farmer.daily_logs}
            return jsonify({"logs": logs_dict})


        # Scenario 3: Save Wizard Settings AND Generate
        elif action == 'save_and_generate':
            settings = data.get('settings')
            if settings:
                if not farmer.alert_settings:
                    farmer.alert_settings = AlertSettings(farmer_id=farmer_id)
                # Map settings to model
                for key, val in settings.items():
                    if hasattr(farmer.alert_settings, key):
                        setattr(farmer.alert_settings, key, val)
                db.session.commit()
                
            advanced_alerts = utils.get_advanced_alerts(settings)
            return jsonify({"alerts": advanced_alerts})
            
        # Scenario 4: Just Generate (e.g. from loaded settings or legacy)
        elif action == 'generate':
            settings = data.get('settings')
            advanced_alerts = utils.get_advanced_alerts(settings)
            return jsonify({"alerts": advanced_alerts})

        # Scenario 5: Legacy Fallback (Direct JSON matching previous behavior)
        else:
            advanced_alerts = utils.get_advanced_alerts(data)
            return jsonify({"alerts": advanced_alerts})

    # GET Request: Prepare initial state
    alert_settings = None
    if farmer.alert_settings:
        alert_settings = {c.name: getattr(farmer.alert_settings, c.name) for c in farmer.alert_settings.__table__.columns}
    
    # Check if daily question answered
    date_key = datetime.now().strftime('%Y-%m-%d')
    answered_today = DailyLog.query.filter_by(farmer_id=farmer_id, date=date_key).count() > 0

    return render_template("alerts.html", locations=locations, alert_settings=alert_settings, answered_today=answered_today)

@app.route("/api/market-prices", methods=["GET"])
def api_market_prices():
    state = request.args.get("state")
    district = request.args.get("district")
    commodity = request.args.get("commodity")
    limit = int(request.args.get("limit", 50))
    data = utils.get_market_prices(state, district, commodity, limit)
    return jsonify(data)

@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def handle_alerts():
    if 'farmer_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    farmer_id = session['farmer_id']
    farmer = Farmer.query.get(farmer_id)
    
    if not farmer:
        return jsonify({"error": "Farmer not found"}), 404
    
    if request.method == "GET":
        alerts = [{
            "id": a.id,
            "commodity": a.commodity,
            "state": a.state,
            "target_price": a.target_price,
            "created_at": a.created_at
        } for a in farmer.price_alerts_list]
        return jsonify(alerts)
        
    if request.method == "POST":
        data = request.json
        new_alert = PriceAlert(
            id=datetime.now().strftime('%Y%m%d%H%M%S'),
            farmer_id=farmer_id,
            commodity=data.get("commodity"),
            state=data.get("state"),
            target_price=int(data.get("target_price")),
            created_at=datetime.now().strftime('%d %b, %Y')
        )
        db.session.add(new_alert)
        db.session.commit()
        return jsonify({
            "status": "success", 
            "alert": {
                "id": new_alert.id,
                "commodity": new_alert.commodity,
                "state": new_alert.state,
                "target_price": new_alert.target_price,
                "created_at": new_alert.created_at
            }
        })
        
    if request.method == "DELETE":
        alert_id = request.args.get("id")
        alert = PriceAlert.query.filter_by(id=alert_id, farmer_id=farmer_id).first()
        if alert:
            db.session.delete(alert)
            db.session.commit()
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Alert not found"}), 404

@app.route("/live-chat")
def live_chat():
    return render_template("live_chat.html")

# --- TUTORIALS ---
@app.route("/tutorials")
def tutorials():
    return render_template("tutorial.html")

@app.route("/tutorials/<path:page_name>")
def tutorial_page(page_name):
    # Dynamic routing for tutorials: tutorials/farming-basics -> tutorial_farming_basics.html
    # This simplifies the many routes in the original file
    template_name = f"tutorial_{page_name.replace('-', '_')}.html"
    try:
        return render_template(template_name)
    except:
        return render_template("tutorial.html") # Fallback

if __name__ == "__main__":
    # Get port from environment variable (default to 5000)
    port = int(os.environ.get("PORT", 5000))
    # Run the app (use_reloader=False stops the duplicate initialization output)
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
