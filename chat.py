import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import pyttsx3
text_speech = pyttsx3.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "MindMate"
print("Let's chat! Type quit to exit")


def getResponse(sentence):
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    print(tag)
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >= 0.60:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                msg = f"<b>{bot_name}</b>: {random.choice(intent['responses'])}"
                return msg


    else:
        msg = f"<b>{bot_name}</b>: I do not understand.. can you please repeat yourself?"
        return msg

