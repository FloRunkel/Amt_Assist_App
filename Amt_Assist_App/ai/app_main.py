from flask import Flask, render_template, request, jsonify
from Chatbot import Chatbot

app = Flask(__name__, template_folder='templates')

data_path = "/Users/florianrunkel/Desktop/Amt_Assist_App/ai/Data_Brain/dialog.csv"
chatbot = Chatbot(data_path)

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/chatbot.html')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json.get('user_input')
    if user_input is None:
        return jsonify({'response': 'Missing user_input field'}), 400
    
    chatbot.load_data()
    chatbot.preprocess_and_tokenize_data() 
    chatbot.split_data()
    chatbot.create_padded_sequences() 
    chatbot.create_label_indices()

    predicted_answer = chatbot.load_and_predict(str(user_input))

    return jsonify({'response': predicted_answer})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
