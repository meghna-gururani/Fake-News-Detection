from flask import Flask, request, render_template, redirect, url_for
import os
import re
import google.generativeai as genai # pip install google-generativeai
from news_detection import get_news_status

app = Flask(__name__) # Initialize the Flask app

app.config['UPLOAD_FOLDER'] = os.getcwd() + '/tmp'
genai.configure(api_key=os.environ['GENAI_API_KEY'])

# Home route - to display the upload form (GET request)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', news_text="", status=None)

# Predict route - to handle the form submission and image processing (POST request)
@app.route('/predict', methods=['POST'])
def process_image():
    if 'input' not in request.files:
        return redirect(url_for('home'))  # Redirect back if no file is uploaded

    # Save image with a numeric filename
    file = request.files['input']
    ext = file.filename.split('.')[-1]
    file_number = len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_number}.{ext}")
    file.save(filepath)
    
    try:
        # Process the image
        sample_file = genai.upload_file(path=filepath)
        os.remove(filepath)
        
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-002") #taken model name from api documentation
        text = "OCR this image:" #optical character recognition (prompt for the model)
        response = model.generate_content([text, sample_file])
        news = response.text.strip().split(':', 1)[-1].strip()  # Extract news content

        # Get the news status (Fake or Real)
        status = get_news_status(news)
        print(f'NEWS: {news}\n{"!!! REAL NEWS !!!" if status else "!!! FAKE NEWS!!!"}')
        return render_template('result.html', news_text=news, status=status)
    
    except Exception as e:
        print(f"ERROR: {e}")
        return render_template('result.html', news_text="Error processing image.", status=None)

if __name__ == '__main__':
    ipv4_add = re.search(r'IPv4 Address.*', os.popen('ipconfig').read()).group().split()[-1]
    app.run(debug=True, host=ipv4_add)
