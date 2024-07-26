
# import json
# import torch
# import random
# import re
# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load general intents
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load faculty classifications
# with open('faculty.json', 'r') as faculty_data:
#     faculty = json.load(faculty_data)

# # Load the trained model
# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Vbot"
# print("Let's chat! (type 'quit' to exit)")

# def classify_faculty(name):
#     """Classify the given faculty name into good, average, or strict."""
#     name = name.strip().title()

#     for category, names in faculty.items():
#         if name in (n.title() for n in names):
#             return f"{name} is {category}."
#     return None


# def classify_faculty_list(input_text):
#     """Classify a list of faculty names from a given text input."""
#     # Normalize and split the input text by commas, new lines, or multiple spaces
#     names = [name.strip().strip('"') for name in re.split(r',\s*|\s{2,}', input_text) if name.strip()]

#     responses = [classify_faculty(name) for name in names]
#     return "\n".join(filter(None, responses)) if responses else "I do not understand..."

# while True:
#     sentence = input("You: ")
#     if sentence.lower() == "quit":
#         break

#     # Classify faculty list
#     response = classify_faculty_list(sentence)
#     if response:
#         print(f"{bot_name}: {response}")
#     else:
#         # Proceed with general intent classification if no faculty name is found
#         sentence_tokens = tokenize(sentence)
#         X = bag_of_words(sentence_tokens, all_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         output = model(X)
#         _, predicted = torch.max(output, dim=1)

#         tag = tags[predicted.item()]

#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     print(f"{bot_name}: {random.choice(intent['responses'])}")
#         else:
#             print(f"{bot_name}: I do not understand...")


import json
import torch
import random
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load general intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load faculty classifications
with open('faculty.json', 'r') as faculty_data:
    faculty = json.load(faculty_data)

# Load the trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Vbot"

def classify_faculty(name):
    """Classify the given faculty name into good, average, or strict."""
    name = name.strip().title()

    for category, names in faculty.items():
        if name in (n.title() for n in names):
            return f"{name} is {category}."
    return None

def classify_faculty_list(input_text):
    """Classify a list of faculty names from a given text input."""
    # Normalize and split the input text by commas, new lines, or multiple spaces
    names = [name.strip().strip('"') for name in re.split(r',\s*|\s{2,}', input_text) if name.strip()]

    responses = [classify_faculty(name) for name in names]
    return "\n".join(filter(None, responses)) if responses else None

def get_response(sentence):
    # Classify faculty list
    response = classify_faculty_list(sentence)
    if response:
        return response
    else:
        # Proceed with general intent classification if no faculty name is found
        sentence_tokens = tokenize(sentence)
        X = bag_of_words(sentence_tokens, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand.I need a concise prompt."
