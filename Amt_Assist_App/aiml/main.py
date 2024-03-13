import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager

from aiml_kernel import aiml_kernel
from chatbot_interface import chatbot_interface

kivy.require('2.0.0')

def get_aiml_files():
    base_path = '/Users/florianrunkel/Desktop/Amt_Assist_App/aiml/standard/'
    file_names = [
        'std-german.aiml',
        'std-welcome.aiml',
        'std-hello.aiml',
        'std-knowledge.aiml',
        'std-robot.aiml',
        'std-sales.aiml',
        'std-sports.aiml',
        'std-srai.aiml',
        'std-suffixes.aiml',
        'std-yesno.aiml',
    ]
    return [base_path + file for file in file_names]

class AIMLChatApp(App):
    def build(self):
        aiml_files = get_aiml_files()
        kernel = aiml_kernel(aiml_files)
        chat_interface = chatbot_interface(kernel)
        return chat_interface

if __name__ == "__main__":
    AIMLChatApp().run()
