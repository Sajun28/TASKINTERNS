import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 3. LSTM Text Generation Model

# 3.1 Data Preparation
text = """
Artificial intelligence (AI) is the simulation of human intelligence processes by machines. 
These processes include learning, reasoning, and self-correction. 
AI applications include expert systems, natural language processing, speech recognition, and machine vision.
Machine learning is a subset of AI that focuses on building systems that learn from data.
Deep learning uses artificial neural networks to mimic the human brain's structure and function.
"""

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# 3.2 Build LSTM Model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 3.3 Train Model
model.fit(X, y, epochs=50, verbose=1)  # Reduce epochs for faster training

# 3.4 Text Generation Function (LSTM)
def generate_text_lstm(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# 4. GPT-2 Text Generation

# 4.1 Load Pre-trained Model
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model.to(device)

# 4.2 Text Generation Function (GPT-2)
def generate_text_gpt(prompt, max_length=100):
    inputs = gpt_tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = gpt_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. User Interaction
def main():
    print("Text Generation System")
    model_choice = input("Choose model (LSTM/GPT): ").upper()
    prompt = input("Enter your topic prompt: ")
    
    if model_choice == "LSTM":
        print("\nGenerating text with LSTM...")
        generated_text = generate_text_lstm(prompt)
    elif model_choice == "GPT":
        print("\nGenerating text with GPT-2...")
        generated_text = generate_text_gpt(prompt)
    else:
        print("Invalid choice! Using GPT-2 as default")
        generated_text = generate_text_gpt(prompt)
    
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
