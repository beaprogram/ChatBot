# ChatBot

An intent based conversational chatbot built with Python, NLTK, and Keras. The bot classifies user input using a trained neural network over a bag of words representation and responds with a context appropriate reply drawn from a configurable intents file. The project demonstrates an end to end natural language processing pipeline from preprocessing through model inference.

## Overview

The chatbot is built around three components: a text preprocessing layer powered by NLTK, a feed forward neural network trained with Keras, and a response engine that maps predicted intents to curated replies. Vocabulary, intent labels, and the trained model are persisted to disk so the runtime stays lightweight and fast.

## Key Features

- Tokenization and lemmatization of user input using NLTK
- Bag of words feature representation built from a learned vocabulary
- Intent classification with a Keras trained neural network
- Confidence thresholding to filter low probability predictions
- Easy customization of conversation behavior through `intents.json`
- Persisted artifacts for vocabulary, classes, and the trained model

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python |
| NLP | NLTK |
| Deep Learning | Keras, TensorFlow |
| Numerical Computing | NumPy |
| Serialization | Pickle, HDF5 |

## Project Structure

```
ChatBot/
chatbot.py              Main inference loop
chatbot/                Training assets and supporting modules
intents.json            Conversation intents, patterns, and responses
words.pkl               Persisted vocabulary
classes.pkl             Persisted intent labels
chatbot_model.h5        Trained Keras model
```

## How It Works

1. User input is tokenized and lemmatized with NLTK.
2. The cleaned tokens are converted into a bag of words vector aligned to the learned vocabulary.
3. The vector is passed through the trained Keras model to produce a probability distribution over intents.
4. Predictions above a configurable confidence threshold are ranked and the top intent is selected.
5. A response is sampled from the matching entry in `intents.json` and printed to the console.

## Getting Started

### Prerequisites

- Python 3.8 or later
- pip

### Installation

```bash
git clone https://github.com/beaprogram/ChatBot.git
cd ChatBot
pip install nltk keras tensorflow numpy
```

Download the required NLTK corpora:

```bash
python -m nltk.downloader punkt wordnet
```

### Running the Chatbot

```bash
python chatbot.py
```

Type a message and press Enter to begin a conversation.

## Customization

To adjust the bot's behavior, edit `intents.json` to add new tags, training patterns, and responses, then retrain the model so the updated intents are reflected in `words.pkl`, `classes.pkl`, and `chatbot_model.h5`.

## Highlights

This project demonstrates a complete machine learning workflow including data preparation, model training, persistence, and real time inference inside a clean Python application.

## Author

Developed by Arup Halder.
