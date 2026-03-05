import os
import base64
import logging
import re
import secrets
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import wikipediaapi
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_FILES_PER_REQUEST = 5

# ============================================================================
# GEMINI API SETUP WITH AUTO-DETECTION
# ============================================================================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("Please set GEMINI_API_KEY in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Auto-detect available vision model
try:
    models = genai.list_models()
    vision_model_name = None
    
    # Look for vision-capable models in order of preference
    preferred_models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro'
    ]
    
    available_models = [m.name for m in models]
    print("✅ Available models:", available_models)
    
    for preferred in preferred_models:
        # Check both with and without 'models/' prefix
        full_name = f"models/{preferred}" if not preferred.startswith('models/') else preferred
        short_name = preferred.replace('models/', '')
        
        if full_name in available_models or short_name in available_models:
            vision_model_name = full_name if full_name in available_models else short_name
            break
    
    if not vision_model_name:
        # Fallback to the most robust model if no matches
        vision_model_name = 'gemini-1.5-flash'
    
    VISION_MODEL = vision_model_name
    vision_model = genai.GenerativeModel(VISION_MODEL)
    print(f"✅ Using model: {VISION_MODEL}")
        
except Exception as e:
    logger.error(f"Model detection error: {e}")
    # Fallback to current best standard model
    VISION_MODEL = 'gemini-1.5-flash'
    vision_model = genai.GenerativeModel(VISION_MODEL)
    print(f"⚠️ Using fallback model: {VISION_MODEL}")

# ============================================================================
# KNOWLEDGE BASE
# ============================================================================
class WildlifeKnowledgeBase:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='WildlifeAI/1.0'
        )
        self.cache = {}
    
    def get_info(self, species_name):
        if not species_name or species_name == "Unknown":
            return {}
        
        cache_key = species_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        info = {
            'common_name': species_name,
            'scientific_name': 'Unknown',
            'conservation_status': 'Not Evaluated',
            'habitat': 'Not available',
            'diet': 'Not available',
            'behavior': 'Not available',
            'fun_facts': []
        }
        
        try:
            page = self.wiki_wiki.page(species_name)
            if page.exists():
                text = page.text[:3000]
                summary = page.summary[:500]
                
                # Scientific name
                sci_match = re.search(r'\(([A-Z][a-z]+ [a-z]+)\)', text[:500])
                if sci_match:
                    info['scientific_name'] = sci_match.group(1)
                
                # Conservation status
                status_patterns = {
                    'Critically Endangered': r'critically endangered|CR',
                    'Endangered': r'endangered|EN',
                    'Vulnerable': r'vulnerable|VU',
                    'Near Threatened': r'near threatened|NT',
                    'Least Concern': r'least concern|LC'
                }
                
                for status, pattern in status_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        info['conservation_status'] = status
                        break
                
                # Fun facts from summary
                facts = summary.split('. ')[:3]
                info['fun_facts'] = [f + '.' for f in facts if f]
                
        except Exception as e:
            logger.error(f"Wiki error: {e}")
        
        self.cache[cache_key] = info
        return info

knowledge_base = WildlifeKnowledgeBase()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_files(files):
    if not files:
        return False, "No files provided"
    
    if len(files) > MAX_FILES_PER_REQUEST:
        return False, f"Maximum {MAX_FILES_PER_REQUEST} files allowed"
    
    for file in files:
        if not allowed_file(file.filename):
            return False, f"Invalid file type: {file.filename}"
        
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {file.filename}"
        
        try:
            img = Image.open(file)
            img.verify()
            file.seek(0)
        except:
            return False, f"Invalid image: {file.filename}"
    
    return True, "Valid files"

def validate_camera_image(image_data):
    """Validate camera captured image"""
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Check size
        if len(image_bytes) > MAX_FILE_SIZE:
            return False, "Image too large"
        
        # Verify image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        
        return True, image_bytes
    except Exception as e:
        logger.error(f"Camera image validation error: {e}")
        return False, str(e)

def create_prompt(language='en'):
    return """You are a wildlife expert. Analyze this image and provide:

=== SPECIES IDENTIFICATION ===
Common Name: [name]
Scientific Name: [name]
Confidence Level: [High/Medium/Low]

=== CONSERVATION STATUS ===
IUCN Status: [status]
Threats: [main threats]

=== HABITAT ===
Primary Habitat: [habitat]
Geographic Range: [range]

=== BIOLOGY ===
Diet: [diet]
Behavior: [behavior]

=== INTERESTING FACTS ===
[3 facts]

If no wildlife, say "No wildlife detected"."""

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if it's a camera capture or file upload
        if request.is_json:
            # Handle camera capture
            data = request.get_json()
            image_data = data.get('image')
            language = data.get('language', 'en')
            user_query = data.get('query', '')[:500]
            
            if not image_data:
                return jsonify({'success': False, 'error': 'No image data provided'}), 400
            
            # Validate camera image
            valid, result = validate_camera_image(image_data)
            if not valid:
                return jsonify({'success': False, 'error': result}), 400
            
            image_bytes = result if isinstance(result, bytes) else base64.b64decode(image_data)
            
            # Process single camera image
            try:
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                prompt = create_prompt(language)
                if user_query:
                    prompt += f"\n\nUser question: {user_query}"
                
                response = vision_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_base64}
                ])
                
                if response and response.text:
                    analysis = response.text
                    
                    # Extract species name
                    species_match = re.search(r'Common Name:\s*([^\n]+)', analysis)
                    species_name = species_match.group(1).strip() if species_match else "Unknown"
                    
                    # Get additional info
                    kb_info = knowledge_base.get_info(species_name)
                    
                    return jsonify({
                        'success': True,
                        'results': [{
                            'success': True,
                            'analysis': analysis,
                            'species_name': species_name,
                            'kb_info': kb_info
                        }]
                    })
                else:
                    return jsonify({'success': False, 'error': 'No response from AI'}), 500
                    
            except Exception as e:
                logger.error(f"Camera analysis error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        else:
            # Handle file upload (existing code)
            if 'images[]' not in request.files:
                return jsonify({'success': False, 'error': 'No images provided'}), 400
            
            files = request.files.getlist('images[]')
            valid, message = validate_files(files)
            if not valid:
                return jsonify({'success': False, 'error': message}), 400
            
            language = request.form.get('language', 'en')
            user_query = request.form.get('query', '')[:500]
            
            results = []
            for file in files:
                try:
                    image_bytes = file.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    prompt = create_prompt(language)
                    if user_query:
                        prompt += f"\n\nUser question: {user_query}"
                    
                    response = vision_model.generate_content([
                        prompt,
                        {"mime_type": "image/jpeg", "data": image_base64}
                    ])
                    
                    if response and response.text:
                        analysis = response.text
                        
                        # Extract species name
                        species_match = re.search(r'Common Name:\s*([^\n]+)', analysis)
                        species_name = species_match.group(1).strip() if species_match else "Unknown"
                        
                        # Get additional info
                        kb_info = knowledge_base.get_info(species_name)
                        
                        results.append({
                            'success': True,
                            'analysis': analysis,
                            'species_name': species_name,
                            'kb_info': kb_info
                        })
                    else:
                        results.append({'success': False, 'error': 'No response'})
                        
                except Exception as e:
                    logger.error(f"Analysis error: {e}")
                    results.append({'success': False, 'error': str(e)})
            
            return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze-camera', methods=['POST'])
def analyze_camera():
    """Dedicated endpoint for camera captures"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        language = data.get('language', 'en')
        user_query = data.get('query', '')[:500]
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode and validate
        try:
            image_bytes = base64.b64decode(image_data)
            
            # Check size
            if len(image_bytes) > MAX_FILE_SIZE:
                return jsonify({'success': False, 'error': 'Image too large'}), 400
            
            # Verify image
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            
            # Process image
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = create_prompt(language)
            if user_query:
                prompt += f"\n\nUser question: {user_query}"
            
            response = vision_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_base64}
            ])
            
            if response and response.text:
                analysis = response.text
                
                # Extract species name
                species_match = re.search(r'Common Name:\s*([^\n]+)', analysis)
                species_name = species_match.group(1).strip() if species_match else "Unknown"
                
                # Get additional info
                kb_info = knowledge_base.get_info(species_name)
                
                return jsonify({
                    'success': True,
                    'analysis': analysis,
                    'species_name': species_name,
                    'kb_info': kb_info
                })
            else:
                return jsonify({'success': False, 'error': 'No response from AI'}), 500
                
        except Exception as e:
            logger.error(f"Camera image processing error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 400
            
    except Exception as e:
        logger.error(f"Camera analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        species_info = data.get('species_info', {})
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2e7d32')
        )
        
        story.append(Paragraph(f"Wildlife Report: {species_info.get('common_name', 'Unknown')}", title_style))
        story.append(Spacer(1, 12))
        
        # Scientific name
        story.append(Paragraph(f"<i>{species_info.get('scientific_name', 'Unknown')}</i>", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Conservation
        story.append(Paragraph(f"<b>Conservation Status:</b> {species_info.get('conservation_status', 'Unknown')}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Habitat
        story.append(Paragraph(f"<b>Habitat:</b> {species_info.get('habitat', 'Unknown')}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Diet
        story.append(Paragraph(f"<b>Diet:</b> {species_info.get('diet', 'Unknown')}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{species_info.get('common_name', 'species')}_report.pdf"
        )
        
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': VISION_MODEL,
        'time': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🦁 WILD LIFE AI - Starting...")
    print("="*60)
    print(f"📷 Model: {VISION_MODEL}")
    print(f"🔑 API Key: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'}")
    print(f"📁 Server: http://localhost:5000")
    print(f"📸 Camera Capture: Supported")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)