from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords


ps=PorterStemmer()

app = Flask(__name__)

# Load model1
model1 = AutoModelForSequenceClassification.from_pretrained("./model1/saved_model")
tokenizer1 = AutoTokenizer.from_pretrained("./model1/saved_model")

# Load model2, model3, model4
with open('./model2/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer2 = pickle.load(vectorizer_file)

with open('./model3/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer3 = pickle.load(vectorizer_file)

with open('./model4/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer4 = pickle.load(vectorizer_file)


with open('./model2/model.pkl', 'rb') as model_file:
    model2 = pickle.load(model_file)

with open('./model3/model.pkl', 'rb') as model_file:
    model3 = pickle.load(model_file)

with open('./model4/model.pkl', 'rb') as model_file:
    model4 = pickle.load(model_file)

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

def predict_model1(text):
    inputs = tokenizer1(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model1(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return predicted_class

def predict_model234(text):
    text_vectorized2 = vectorizer2.transform([text])
    text_vectorized3 = vectorizer3.transform([text])
    text_vectorized4 = vectorizer4.transform([text])


    predicted_class2 = model2.predict(text_vectorized2)[0]
    predicted_class3 = model3.predict(text_vectorized3)[0]
    predicted_class4 = model4.predict(text_vectorized4)[0]


    return predicted_class2, predicted_class3, predicted_class4

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        print(text)
        text=transform_text(text)
        print(text)

        result_model1 = predict_model1(text)

        print("Model 1 = ",result_model1)

        # if result_model1 == 1:
        if True:
            result_model2, result_model3, result_model4 = predict_model234(text)


            response = {
                "model2": str(result_model2),
                "model3": str(result_model3),
                "model4": str(result_model4),
                "severity":1
            }
        else:
            response = {"message": "Not applicable"}

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
