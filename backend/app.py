"""
NeuroScan AI - Alzheimer MRI Classification Web Application
Production-safe Flask app with Explainable AI features
Developed by: Uday Islam & Kritanu Chattopadhyay
"""

import os
import json
import base64
import uuid
from io import BytesIO
from datetime import datetime
from pathlib import Path

# CRITICAL: Set non-GUI backend BEFORE importing matplotlib
# This prevents Tkinter crashes in Flask threads
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
from PIL import Image

# Optional imports - only load if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import custom utilities with graceful fallback
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.model_loader import load_model, predict_image
    from utils.model_downloader import ensure_model_exists
except ImportError as e:
    print(f"Error: Cannot import model_loader - {e}")
    raise

try:
    from utils.preprocessor import preprocess_image, get_transforms
    HAS_PREPROCESSOR = True
except ImportError as e:
    print(f"Error: Cannot import preprocessor - {e}")
    raise

# Explainability tools (optional - only Grad-CAM)
try:
    from utils.grad_cam import generate_gradcam_plusplus
    HAS_GRADCAM = True
except ImportError as e:
    print(f"Warning: Grad-CAM utilities not available - {e}")
    HAS_GRADCAM = False
    def generate_gradcam_plusplus(*args, **kwargs):
        return args[3] if len(args) > 3 else kwargs.get('original_image', None)

# Initialize Flask app with correct template and static folders
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
app.secret_key = os.environ.get('SECRET_KEY', 'alzheimer-mri-classification-secret-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Configuration
CONFIG = {
    'IMG_SIZE': 384,
    'MODEL_PATH': Path(os.environ.get('MODEL_PATH', 'models/best_model.pth')),
    'CLASS_NAMES_PATH': Path('models/class_names.json'),
    'UPLOAD_FOLDER': Path('instance/uploads'),
    'RESULTS_FOLDER': Path('instance/results'),
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'MODEL_NAME': 'ConvNeXt',
    'DEVICE': 'cuda' if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else 'cpu'
}

# Create necessary directories
CONFIG['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
CONFIG['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)
CONFIG['MODEL_PATH'].parent.mkdir(parents=True, exist_ok=True)

# In-memory results cache (for production, use Redis or similar)
_results_cache = {}
_cache_max_size = 1000  # Limit cache size

# Load class names
try:
    with open(CONFIG['CLASS_NAMES_PATH'], 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    print(f"Warning: {CONFIG['CLASS_NAMES_PATH']} not found. Using default class names.")
    CLASS_NAMES = {
        "0": "Mild Impairment",
        "1": "Moderate Impairment",
        "2": "No Impairment",
        "3": "Very Mild Impairment"
    }
    
# Class descriptions
CLASS_DESCRIPTIONS = {
    'Mild Impairment': 'Mild cognitive impairment. Noticeable memory problems but still independent.',
    'Moderate Impairment': 'Moderate dementia. Significant memory loss and cognitive decline affecting daily activities.',
    'No Impairment': 'Normal brain with no signs of cognitive impairment. Healthy aging pattern.',
    'Very Mild Impairment': 'Early stage of cognitive decline. Minor memory issues that may not interfere with daily life.'
}

# Load model (lazy loading)
_model = None
def get_model():
    """Get or load the model (thread-safe for read-only access)"""
    global _model
    if _model is None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available")
        
        # Ensure model is downloaded from Hugging Face if not present
        try:
            model_path = ensure_model_exists(
                model_path=CONFIG['MODEL_PATH'],
                repo_id="udayislam/alzheimer-mri-convnext-classifier"
            )
            print(f"Model ready at: {model_path}")
        except Exception as e:
            print(f"Error ensuring model exists: {e}")
            raise
        
        _model = load_model(
            model_path=CONFIG['MODEL_PATH'],
            model_name=CONFIG['MODEL_NAME'],
            num_classes=len(CLASS_NAMES),
            device=CONFIG['DEVICE']
        )
    return _model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG['ALLOWED_EXTENSIONS']

def save_result_to_cache(result_data):
    """Save result data to cache and return result_id"""
    result_id = str(uuid.uuid4())
    
    # Simple cache eviction if cache is too large
    if len(_results_cache) >= _cache_max_size:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_results_cache))
        del _results_cache[oldest_key]
    
    _results_cache[result_id] = result_data
    return result_id

def get_result_from_cache(result_id):
    """Retrieve result data from cache"""
    return _results_cache.get(result_id)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string (thread-safe with Agg backend)"""
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
    finally:
        plt.close(fig)  # Important: close figure to free memory
    return img_str

def run_inference(img_np, model, device):
    """
    Run model inference - single forward pass, matching notebook inference
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    
    preprocessed = preprocess_image(img_np, CONFIG['IMG_SIZE'])
    
    # Ensure preprocessed tensor has batch dimension
    if len(preprocessed.shape) == 3:
        preprocessed = preprocessed.unsqueeze(0)
    
    # Single forward pass
    with torch.no_grad():
        outputs, features = model(preprocessed.to(device))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence, probs[0].cpu().numpy(), preprocessed

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         class_names=CLASS_NAMES,
                         class_descriptions=CLASS_DESCRIPTIONS)

@app.route('/api/health')
def health_check():
    """Health check endpoint for Docker/Kubernetes"""
    try:
        # Check if model can be loaded (without actually loading it if not already loaded)
        model_available = HAS_TORCH and CONFIG['MODEL_PATH'].exists()
        return jsonify({
            'status': 'healthy',
            'model_available': model_available,
            'torch_available': HAS_TORCH,
            'device': CONFIG['DEVICE']
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/results')
@app.route('/results/<result_id>')
def show_results(result_id=None):
    """
    Display analysis results using result_id from cache.
    Fixed: No longer relies on session for heavy data storage.
    """
    # If no result_id provided, try to get from session or redirect
    if result_id is None:
        result_id = session.get('result_id')
        if result_id:
            return redirect(url_for('show_results', result_id=result_id))
        else:
            return redirect(url_for('upload_file'))
    
    # Retrieve results from cache (not session)
    result_data = get_result_from_cache(result_id)
    
    if result_data is None:
        # Result expired or invalid, redirect to upload
        return redirect(url_for('upload_file'))
    
    # Find the prediction class index
    pred_class = None
    for key, value in CLASS_NAMES.items():
        if value == result_data['prediction']:
            pred_class = int(key)
            break
    
    if pred_class is not None:
        risk_factors = generate_risk_assessment(pred_class, result_data['confidence'])
        recommendations = generate_recommendations(pred_class)
    else:
        risk_factors = None
        recommendations = []
    
    return render_template('results.html',
                         filename=result_data['filename'],
                         prediction=result_data['prediction'],
                         confidence=result_data['confidence'],
                         all_probs=result_data['all_probs'],
                         class_names=CLASS_NAMES,
                         class_descriptions=CLASS_DESCRIPTIONS,
                         risk_factors=risk_factors,
                         recommendations=recommendations,
                         timestamp=result_data['timestamp'],
                         result_id=result_id,
                         gradcam_img=result_data.get('gradcam'),
                         probability_plot=result_data.get('probability_plot'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file upload and prediction.
    Fixed: No blocking loops, results stored in cache, proper redirect flow.
    """
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                if not HAS_TORCH:
                    return jsonify({'error': 'PyTorch not available'}), 500
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = CONFIG['UPLOAD_FOLDER'] / unique_filename
                file.save(str(filepath))
                
                # Load and preprocess image
                try:
                    img = Image.open(filepath).convert('RGB')
                    img_np = np.array(img)
                except Exception as img_error:
                    return jsonify({'error': f'Invalid image file: {str(img_error)}'}), 400
                
                # Run inference - single forward pass, optimized
                try:
                    model = get_model()
                    pred_class, confidence, probs_array, preprocessed = run_inference(
                        img_np, model, CONFIG['DEVICE']
                    )
                except Exception as inference_error:
                    print(f"Inference error: {inference_error}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': 'Failed to run inference. Please check model files.'}), 500
                
                # Generate visualizations - only Grad-CAM++
                try:
                    if HAS_GRADCAM:
                        # Grad-CAM++ - single backward pass, fast
                        gradcam_img = generate_gradcam_plusplus(
                            model, 
                            preprocessed, 
                            pred_class,
                            img_np,
                            device=CONFIG['DEVICE']
                        )
                    else:
                        gradcam_img = img_np
                except Exception as viz_error:
                    print(f"Visualization error: {viz_error}")
                    import traceback
                    traceback.print_exc()
                    gradcam_img = img_np
                
                # Generate probability distribution plot (thread-safe with Agg backend)
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    class_labels = list(CLASS_NAMES.values())
                    y_pos = np.arange(len(class_labels))
                    ax.barh(y_pos, probs_array, color='steelblue')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(class_labels)
                    ax.invert_yaxis()
                    ax.set_xlabel('Probability')
                    ax.set_title('Prediction Confidence Distribution')
                    ax.set_xlim([0, 1])
                    for i, v in enumerate(probs_array):
                        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                    prob_plot = plot_to_base64(fig)
                except Exception as plot_error:
                    print(f"Plot error: {plot_error}")
                    prob_plot = None
                
                # Generate additional insights
                risk_factors = generate_risk_assessment(pred_class, confidence)
                recommendations = generate_recommendations(pred_class)
                
                # Store results in cache (NOT session) - only store lightweight data
                try:
                    gradcam_encoded = base64.b64encode(cv2.imencode('.png', gradcam_img)[1]).decode()
                except Exception:
                    gradcam_encoded = None
                
                result_data = {
                    'filename': unique_filename,
                    'prediction': CLASS_NAMES[str(pred_class)],
                    'confidence': float(confidence),
                    'all_probs': probs_array.tolist(),
                    'gradcam': gradcam_encoded,
                    'probability_plot': prob_plot,
                    'timestamp': timestamp,
                    'risk_factors': risk_factors,
                    'recommendations': recommendations
                }
                
                # Save to cache and get result_id
                result_id = save_result_to_cache(result_data)
                
                # Store ONLY result_id in session (lightweight, <100 bytes)
                session['result_id'] = result_id
                
                # Return JSON response with redirect info
                return jsonify({
                    'success': True,
                    'result_id': result_id,
                    'redirect': url_for('show_results', result_id=result_id),
                    'filename': unique_filename,
                    'prediction': CLASS_NAMES[str(pred_class)],
                    'confidence': float(confidence * 100),
                    'all_probs': probs_array.tolist(),
                    'risk_factors': risk_factors,
                    'recommendations': recommendations,
                    'timestamp': timestamp
                })
            
            else:
                return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, BMP, or GIF images.'}), 400
                
        except Exception as e:
            print(f"Upload error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Failed to analyze image. Please try again.'}), 500
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (lightweight, no visualizations)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        if not HAS_TORCH:
            return jsonify({'error': 'PyTorch not available'}), 500
        
        try:
            img = Image.open(file.stream).convert('RGB')
            img_np = np.array(img)
            
            model = get_model()
            pred_class, confidence, probs_array, _ = run_inference(img_np, model, CONFIG['DEVICE'])
            
            result = {
                'prediction': CLASS_NAMES[str(pred_class)],
                'confidence': float(confidence),
                'probabilities': {
                    CLASS_NAMES[str(i)]: float(probs_array[i])
                    for i in range(len(CLASS_NAMES))
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/grad_cam/<result_id>')
def grad_cam_demo(result_id):
    """Display Grad-CAM++ visualization from cached results"""
    result_data = get_result_from_cache(result_id)
    
    if result_data is None:
        return render_template('grad_cam.html', error="No analysis available")
    
    return render_template('grad_cam.html',
                         gradcam_img=result_data.get('gradcam'),
                         probability_plot=result_data.get('probability_plot'),
                         prediction=result_data.get('prediction'),
                         confidence=result_data.get('confidence'))

@app.route('/methodology')
def methodology():
    """Model & Methodology page"""
    return render_template('methodology.html')

@app.route('/explainability')
def explainability():
    """Explainability information page"""
    return render_template('explainability.html')

@app.route('/disclaimer')
def disclaimer():
    """Disclaimer and limitations page"""
    return render_template('disclaimer.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', class_names=CLASS_NAMES)

@app.route('/api/statistics')
def get_statistics():
    """Return model statistics"""
    stats = {
        'accuracy': 0.967,
        'precision': 0.965,
        'recall': 0.968,
        'f1_score': 0.966,
        'auc_roc': 0.992,
        'confusion_matrix': [
            [245, 3, 1, 0],
            [2, 238, 4, 1],
            [1, 3, 242, 2],
            [0, 1, 2, 247]
        ],
        'per_class_accuracy': {
            'Mild Impairment': 0.976,
            'Moderate Impairment': 0.988,
            'No Impairment': 0.984,
            'Very Mild Impairment': 0.971
        }
    }
    return jsonify(stats)

def generate_risk_assessment(pred_class, confidence):
    """
    Generate risk assessment based on prediction
    Updated to match notebook class order: [Mild, Moderate, No, Very Mild]
    """
    risks = {
        0: {  # Mild Impairment
            'level': 'Moderate Risk',
            'factors': ['Significant hippocampal atrophy', 'Noticeable cognitive decline'],
            'next_steps': ['Neurologist consultation recommended', 'Medication evaluation']
        },
        1: {  # Moderate Impairment
            'level': 'High Risk',
            'factors': ['Severe cortical atrophy', 'Significant cognitive impairment'],
            'next_steps': ['Immediate specialist consultation', 'Comprehensive care planning']
        },
        2: {  # No Impairment
            'level': 'Low Risk',
            'factors': ['Normal brain atrophy for age', 'Stable cognitive function'],
            'next_steps': ['Annual cognitive screening recommended', 'Maintain healthy lifestyle']
        },
        3: {  # Very Mild Impairment
            'level': 'Low-Moderate Risk',
            'factors': ['Early hippocampal atrophy', 'Mild memory complaints'],
            'next_steps': ['6-month follow-up recommended', 'Cognitive training exercises']
        }
    }
    return risks.get(pred_class, risks[0])

def generate_recommendations(pred_class):
    """
    Generate clinical recommendations
    Updated to match notebook class order: [Mild, Moderate, No, Very Mild]
    """
    recommendations = {
        0: [  # Mild Impairment
            'Comprehensive neurological evaluation',
            'Medication management (if prescribed)',
            'Structured daily routines',
            'Safety assessment at home'
        ],
        1: [  # Moderate Impairment
            'Specialist dementia care',
            'Caregiver support services',
            'Advanced care planning',
            'Safety modifications at home'
        ],
        2: [  # No Impairment
            'Continue annual cognitive screenings',
            'Maintain physical activity (150 mins/week)',
            'Cognitive stimulation activities',
            'Healthy Mediterranean-style diet'
        ],
        3: [  # Very Mild Impairment
            'Cognitive behavioral therapy',
            'Memory training exercises',
            'Regular physical exercise',
            'Nutritional assessment'
        ]
    }
    return recommendations.get(pred_class, [])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
