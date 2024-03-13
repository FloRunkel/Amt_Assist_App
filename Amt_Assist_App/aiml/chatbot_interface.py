from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

class ChatBox(BoxLayout):
    def __init__(self, text, background_color=(0.95, 0.95, 0.95, 1), **kwargs):  
        super().__init__(**kwargs)
        self.size_hint_y = None  
        self.padding = (10, 5)
        self.background_color = background_color

        self.message_text = TextInput(text=text, multiline=True, readonly=True, foreground_color=(0, 0, 0, 1), font_size=20)
        self.add_widget(self.message_text)
        

class chatbot_interface(GridLayout):
    def __init__(self, kernel, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1
        self.padding = [10, 10, 10, 10]
        self.spacing = 10
        self.kernel = kernel

        self.chat_history = ScrollView(always_overscroll=True)  
        self.chat_history_box = BoxLayout(orientation='vertical')
        self.chat_history.add_widget(self.chat_history_box)
        self.add_widget(self.chat_history)

        self.create_user_input()

    def create_user_input(self):
        user_input_layout = BoxLayout(orientation='horizontal')
        Window.clearcolor = (1, 1, 1, 1)  

        self.user_input = TextInput(background_color=(1, 1, 1, 1), foreground_color=(0, 0, 0, 1), size_hint=(0.8, 0.2))
        user_input_layout.add_widget(self.user_input)

        send_button = Button(text="Send", background_color=(0.7, 0.8, 1, 1), color=(1, 1, 1, 1), size_hint=(0.2, 0.2))  
        send_button.bind(on_press=self.send_message)
        user_input_layout.add_widget(send_button)

        self.add_widget(user_input_layout)

    def send_message(self, instance):
        user_input = self.user_input.text
        self.user_input.text = ''

        if user_input.lower() == "exit":
            App.get_running_app().stop()
        else:
            response = self.kernel.respond(user_input)
            self.update_chat_history(f"You: {user_input}\nBot: {response}\n")

    def update_chat_history(self, message):
        if message.startswith("You:"):
            chat_box = ChatBox(message, background_color=(1, 1, 1, 1))  
        else:
            chat_box = ChatBox(message, background_color=(1, 1, 1, 1))  
        self.chat_history_box.add_widget(chat_box)
        self.chat_history.scroll_to(chat_box, padding=10)

Window.size = (400, 600)  
