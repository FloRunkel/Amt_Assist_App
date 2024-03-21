from Chatbot import Chatbot

data_path = "/Users/florianrunkel/Desktop/Amt_Assist_App/ai/Data_Brain/dialog.csv"
chatbot = Chatbot(data_path)
chatbot.load_data()
chatbot.preprocess_and_tokenize_data() 
chatbot.split_data()
chatbot.create_padded_sequences() 
chatbot.create_label_indices()
#chatbot.train_model()
#chatbot.save_model()
#chatbot.evaluation()
chatbot.load_and_predict('Hi')
