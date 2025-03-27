import os
import pandas as pd
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize AI model (DialoGPT)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for chat and file processing
@app.route('/chat', methods=['POST'])
def chat():
    message = request.form.get("message")
    file = request.files.get("file")
    response_text = ""

    # File handling
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the file based on its type
        try:
            if filename.endswith('.xlsx'):
                data = pd.read_excel(filepath)
                response_text += "Extracted Excel Data:\n" + data.to_string()
            elif filename.endswith('.pdf'):
                doc = fitz.open(filepath)
                pdf_text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pdf_text += page.get_text()
                response_text += "Extracted PDF Text:\n" + pdf_text
            else:
                response_text += "File uploaded but type not fully supported."
        except Exception as e:
            return jsonify({"reply": f"Error processing file: {str(e)}"})

    # Chatbot response
    if message:
        ai_response = chatbot(message, max_length=50, num_return_sequences=1)[0]['generated_text']
        response_text += "\n\nAI Response: " + ai_response

    return jsonify({"reply": response_text})

if __name__ == "__main__":
    app.run(debug=True)

