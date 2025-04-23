from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import math
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text):
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Calculate TF
    word_counts = Counter(words)
    total_words = len(words)
    tf = {word: count/total_words for word, count in word_counts.items()}
    
    # Calculate IDF (using 1 as document count since we're processing a single document)
    idf = {word: math.log(1/1) for word in word_counts.keys()}
    
    # Combine results
    results = []
    for word in word_counts.keys():
        results.append({
            'word': word,
            'tf': tf[word],
            'idf': idf[word]
        })
    
    # Sort by IDF in descending order and get top 50
    results.sort(key=lambda x: x['idf'], reverse=True)
    return results[:50]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
        results = process_text(text)
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 