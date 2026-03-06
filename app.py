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

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app)

# =========================================================
# CONFIG
# =========================================================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 5

# =========================================================
# GEMINI API
# =========================================================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Stable vision model
VISION_MODEL = "gemini-1.5-flash"

vision_model = genai.GenerativeModel(VISION_MODEL)

print("✅ Gemini Model:", VISION_MODEL)

# =========================================================
# KNOWLEDGE BASE
# =========================================================
class WildlifeKnowledgeBase:

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='WildlifeAI/1.0'
        )
        self.cache = {}

    def get_info(self, species_name):

        if not species_name or species_name == "Unknown":
            return {}

        key = species_name.lower()

        if key in self.cache:
            return self.cache[key]

        info = {
            "common_name": species_name,
            "scientific_name": "Unknown",
            "conservation_status": "Unknown",
            "habitat": "Unknown",
            "diet": "Unknown",
            "behavior": "Unknown",
            "fun_facts": []
        }

        try:
            page = self.wiki.page(species_name)

            if page.exists():

                text = page.text[:2000]

                sci = re.search(r'\(([A-Z][a-z]+ [a-z]+)\)', text)

                if sci:
                    info["scientific_name"] = sci.group(1)

                facts = page.summary.split(". ")[:3]

                info["fun_facts"] = [f + "." for f in facts]

        except Exception as e:
            logger.error(e)

        self.cache[key] = info

        return info


knowledge_base = WildlifeKnowledgeBase()

# =========================================================
# HELPERS
# =========================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_files(files):

    if len(files) > MAX_FILES_PER_REQUEST:
        return False, "Too many files"

    for file in files:

        if not allowed_file(file.filename):
            return False, "Invalid file type"

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > MAX_FILE_SIZE:
            return False, "File too large"

    return True, "ok"


def create_prompt():

    return """
You are a wildlife expert.

Analyze this image and provide:

Common Name:
Scientific Name:
Confidence Level:

Habitat:
Diet:
Behavior:

Conservation Status:

3 Interesting Facts
"""


# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template("index.html")


# =========================================================
# IMAGE ANALYSIS
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if 'images[]' not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"})

        files = request.files.getlist("images[]")

        valid, msg = validate_files(files)

        if not valid:
            return jsonify({"success": False, "error": msg})

        results = []

        for file in files:

            image_bytes = file.read()

            img = Image.open(io.BytesIO(image_bytes))

            prompt = create_prompt()

            response = vision_model.generate_content([prompt, img])

            analysis = response.text

            species_match = re.search(r'Common Name:\s*([^\n]+)', analysis)

            species = species_match.group(1).strip() if species_match else "Unknown"

            kb = knowledge_base.get_info(species)

            results.append({
                "success": True,
                "analysis": analysis,
                "species_name": species,
                "kb_info": kb
            })

        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as e:

        logger.error(e)

        return jsonify({
            "success": False,
            "error": str(e)
        })


# =========================================================
# CAMERA IMAGE
# =========================================================

@app.route("/analyze-camera", methods=["POST"])
def analyze_camera():

    try:

        data = request.get_json()

        image_data = data.get("image")

        if not image_data:
            return jsonify({"success": False, "error": "No image"})


        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        img = Image.open(io.BytesIO(image_bytes))

        prompt = create_prompt()

        response = vision_model.generate_content([prompt, img])

        analysis = response.text

        species_match = re.search(r'Common Name:\s*([^\n]+)', analysis)

        species = species_match.group(1).strip() if species_match else "Unknown"

        kb = knowledge_base.get_info(species)

        return jsonify({
            "success": True,
            "analysis": analysis,
            "species_name": species,
            "kb_info": kb
        })

    except Exception as e:

        logger.error(e)

        return jsonify({
            "success": False,
            "error": str(e)
        })


# =========================================================
# PDF DOWNLOAD
# =========================================================

@app.route("/download-pdf", methods=["POST"])
def download_pdf():

    data = request.get_json()

    species = data.get("species_info", {})

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()

    story = []

    title_style = ParagraphStyle(
        'title',
        parent=styles['Heading1'],
        textColor=colors.green
    )

    story.append(Paragraph(
        f"Wildlife Report: {species.get('common_name','Unknown')}",
        title_style
    ))

    story.append(Spacer(1, 20))

    story.append(Paragraph(
        f"Scientific Name: {species.get('scientific_name','Unknown')}",
        styles['Normal']
    ))

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Conservation Status: {species.get('conservation_status','Unknown')}",
        styles['Normal']
    ))

    doc.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="wildlife_report.pdf",
        mimetype="application/pdf"
    )


# =========================================================
# HEALTH
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model": VISION_MODEL,
        "time": datetime.now().isoformat()
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("=================================")
    print("Wildlife AI Server Starting")
    print("Model:", VISION_MODEL)
    print("=================================")

    app.run(host="0.0.0.0", port=5000)
