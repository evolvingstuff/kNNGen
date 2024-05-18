import pickle
import random
import time
import tqdm
import faiss
import torch
import numpy as np
import logging
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM)


# TODO move to config file
K = 1
model_name = 'gpt2'
dataset_config = {
    "path": "wikitext",
    "name": "wikitext-2-raw-v1"
}


def get_embeddings_and_tokens(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)  # TODO
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # Remove batch dimension
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    assert len(embeddings) == len(tokens), 'embeddings/tokens mismatch'
    return embeddings, tokens


def get_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def create_faiss_index(model, tokenizer, corpus):
    d = model.config.n_embd
    index = faiss.IndexFlatL2(d)
    all_embeddings = []
    all_tokens = []
    for text in tqdm.tqdm(corpus):
        embeddings, tokens = get_embeddings_and_tokens(model, tokenizer, text)
        for i in range(len(tokens) - 1):
            all_embeddings.append(embeddings[i])
            all_tokens.append(tokens[i+1])
    embeddings_array = np.array(all_embeddings, dtype='float32')
    index.add(embeddings_array)
    return index, all_tokens, embeddings_array


def interactive_text_generator(model, tokenizer, index, tokens):
    print("Interactive Text Generator:")
    while True:
        input_text = input("Enter your text (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        tokens_to_generate = int(input("Tokens: "))
        k = int(input("K: "))
        generated_text = input_text
        for t in range(tokens_to_generate):
            next_token = predict_next_token(generated_text, model, tokenizer, index, tokens, k)
            generated_text = generated_text + next_token
        print(generated_text)


def predict_next_token(input_text, model, tokenizer, index, tokens, k):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    assert len(inputs) == len(outputs), 'output output mismatch'
    hidden_state_at_position = outputs.last_hidden_state[:, -1, :].numpy()
    distances, indices = index.search(hidden_state_at_position, k)
    predicted_tokens = [tokens[i] for i in indices[0]]
    next_token = random.choice(predicted_tokens)
    next_token = next_token.replace("Ġ", " ")
    return next_token


def main():
    logging.basicConfig(level=logging.INFO)

    torch.set_num_threads(1)

    print(f'preparing {model_name} model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print(f'done preparing {model_name} model.')

    print(f'preparing {dataset_config["name"]} dataset...')
    dataset = load_dataset(**dataset_config)
    print(dataset)

    corpus = []

    for example in dataset['train']:
        txt = example['text'].strip()
        if txt == '':
            continue
        print(txt)
        corpus.append(txt)

    if os.path.exists('data.pkl'):
        print('loading index from pkl..')
        with open('data.pkl', 'rb') as f:
            (index, tokens, embeddings) = pickle.load(f)
    else:
        print('creating index...')
        t1 = time.time()
        index, tokens, embeddings = create_faiss_index(model, tokenizer, corpus)
        t2 = time.time()
        print(f'processing corpus took {(t2 - t1):.2f} seconds.')
        with open('data.pkl', 'wb') as f:
            pickle.dump((index, tokens, embeddings), f)

    # Start the interactive session
    interactive_text_generator(model, tokenizer, index, tokens)


if __name__ == '__main__':
    main()
