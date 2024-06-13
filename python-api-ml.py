import random
import json
import torch
import nltk
nltk.download('punkt')
from flask import Flask, jsonify, request
import torch.nn.functional as F
import requests
from pyngrok import ngrok
import torch.nn as nn
import numpy as np
from nltk.stem.porter import PorterStemmer
from io import BytesIO  # Missing import
import os
import google.generativeai as genai
stemmer = PorterStemmer()

port_no = 5000

app = Flask(__name__)

ngrok.set_auth_token('NGROK_AUTH_TOKEN')
public_url = ngrok.connect(port_no).public_url

INTENTS_FILE_PATH = 'https://test.onlinedubaivisas.com/chatbot/intents.json'

GEMINI_API_KEY = "AIzaSyDK-VrYB-xeK62LM4NaxwaG7jCglMADfyc"

genai.configure(
	api_key=GEMINI_API_KEY
)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

@app.route('/')
def home():
    return "Hello, Welcome to Sethma Hospital!"

@app.route('/api/update-dataset')
def datasets():
# URL of the server file
    url = "https://test.onlinedubaivisas.com/chatbot/data.pth"

    try:
        # Fetch the file from the server
        response = requests.get(url)

        # Raise an exception for HTTP errors
        response.raise_for_status()

        # Load the file into a BytesIO object
        file_bytes = BytesIO(response.content)

        # Use torch.load to load the file
        data = torch.load(file_bytes)

        # Print to verify the data
        #return("Data loaded successfully:")
        response_data = {
            'message_type': 'successfully',
            'message': 'Data loaded successfully',
        }
        return jsonify(response_data)
        #print(data)

    except requests.exceptions.RequestException as e:
        response_data = {
            'message_type': 'error',
            'message': f"Error fetching the file: {e}",
        }
        return jsonify(response_data)
        #return(f"Error fetching the file: {e}")
    except Exception as e:
        response_data = {
            'message_type': 'error',
            'message': f"Error loading the file with torch: {e}",
        }
        return jsonify(response_data)
        #return(f"Error loading the file with torch: {e}")

@app.route("/api")
def message_api():
    return "Welcome to Sethma Hospital, please put your message to here!!"

@app.route("/api/<sentence>")
def get_message(sentence):
    if sentence:
        def tokenize(sentence):
            return nltk.word_tokenize(sentence)

        def stem(word):
            return stemmer.stem(word.lower())

        def bag_of_words(tokenized_sentence, all_words):
            tokenized_sentence = [stem(w) for w in tokenized_sentence]
            bag = np.zeros(len(all_words), dtype=np.float32)
            for index, word in enumerate(all_words):
                if word in tokenized_sentence:
                    bag[index] = 1.0
            return bag

        class NuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NuralNet, self).__init__()
                self.l1 = nn.Linear(input_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.l3 = nn.Linear(hidden_size, num_classes)
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.l1(x)
                out = self.relu(out)
                out = self.l2(out)
                out = self.relu(out)
                out = self.l3(out)
                return out

        server_json_url = 'https://test.onlinedubaivisas.com/chatbot/intents.json'
        response = requests.get(server_json_url)
        response.raise_for_status()
        intents = response.json()

        url = "https://test.onlinedubaivisas.com/chatbot/data.pth"
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = BytesIO(response.content)
        data = torch.load(file_bytes)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
        model_state = data["model_state"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        sentence_tokens = tokenize(sentence)
        X = bag_of_words(sentence_tokens, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probability = torch.softmax(output, dim=1)
        probability_actual = probability[0][predicted.item()]
        bot_response = "I don't understand your question..."
        #response_data = {
        #    'chat_message': sentence,
        #    'bot_message': probability_actual.item()
        #}
        #return jsonify(response_data)
        if probability_actual.item() >= 0.999977:
            for intent in intents["intents"]:
                #if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])

                response_data = {
                    'chat_message': sentence,
                    'bot_message': bot_response,
                    'chat_type' : 'json'
                }

                return jsonify(response_data)

                #else:
                #    bot_response = "I don't understand your question..."
                #    response_data = {
                #        'chat_message': sentence,
                #        'bot_message': tag+'-'+intent["tag"],
                #        'chat_type' : 'own'
                #    }

                #    return jsonify(response_data)

        else:
          response = chat.send_message(sentence)
          response_data = {
              'chat_message': sentence,
              'bot_message': response.text,
              'chat_type' : 'gemini'
          }

          #new_intent = {
          #    "tag": tag,
          #    "patterns": [sentence],
          #    "responses": [response.text]
          #}
          #intents["intents"].append(new_intent)

          # Save the updated intents to the file
          #with open(INTENTS_FILE_PATH, 'w') as f:
          #    json.dump(intents, f, indent=4)

          return jsonify(response_data)

    else:
        return "Please Enter your message here!!"

print(f"To access the Global Link Please Click {public_url}")

app.run(port=port_no)
