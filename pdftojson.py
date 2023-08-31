

import PyPDF2
import json
from nltk import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''

        NumPages = len(reader.pages)

        for i in range(0, NumPages):
            PageObj = reader.pages[i]
            print("this is page " + str(i)) 
            text += PageObj.extract_text() 
        return text

def perform_semantic_analysis(text):
    sentences = sent_tokenize(text)
    sid = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        sentiment_scores = sid.polarity_scores(sentence)
        result = {
            'sentence': sentence,
            'tokens': tokens,
            'sentiment': sentiment_scores['compound']
        }
        results.append(result)
    return results

def generate_json_file(data, json_path):
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
pdf_path = 'C:\\Users\\HP\\Desktop\\intern-pyth\\AltoK10.pdf'
json_path = 'C:\\Users\\HP\\Desktop\\intern-pyth\\extract2.json'

text = extract_text_from_pdf(pdf_path)
results = perform_semantic_analysis(text)
generate_json_file(results, json_path)
