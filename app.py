from flask import Flask , render_template, request
import string
import numpy as numpy
import nltk
from nltk.stem.porter import PorterStemmer
import joblib
import os


#punctuation = string.punctuation
stop_words = nltk.corpus.stopwords.words('english')
stemmer = PorterStemmer()



model = joblib.load(r'H:\Projects\NLP\Fake News\fullPipline.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    output = ""
    data = request.form['message']
    data = data.lower()
    data = "".join([word for word in data if word not in punctuation])
    print(data)
    data = nltk.tokenize.word_tokenize(data)
    data = [word for word in data if word not in stop_words]
    data = [stemmer.stem(word) for word in data]
    data = " ".join(data)
    dateList = [data]
    prediction = model.predict(dateList)
    if prediction[0] == 1:
        output = "The article if Fake"
    else:
        output = "The article is not Fake"
    return render_template('home.html',prediction_text=output)


if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=True)

