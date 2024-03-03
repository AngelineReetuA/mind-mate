from nltk_utils import stem, tokenize, bag_of_words
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
with open('intents.json', 'r') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['.', '/', '?', ',', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # cross entropy loss

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = torch.from_numpy(y_train).long()


class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 850

# print(input_size, len(all_words))
# print(output_size, tags)
if __name__ == '__main__':
    dataset = chatDataset()
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}')
        total_loss = 0.0
        total_batches = 0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / total_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training completed, File saved to {FILE}')
