import pickle
import random
import time
from dotenv import load_dotenv
import tqdm
import torch
import numpy as np
import logging
import os
# avoid potential OpenMP version conflicts between huggingface and faiss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import faiss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM)


# TODO move to config file
K = 1
device_name = 'mps'  # 'mps'
space_token = '▁'  # "Ġ"
model_name = 'mistralai/Mistral-7B-v0.1'  # 'gpt2'
dataset_config = {
    "path": "wikitext",
    "name": "wikitext-2-raw-v1"
}
use_quantization = True


def create_faiss_index(model, tokenizer, corpus, device):
    hidden_dim = model.config.hidden_size  # model.config.n_embd
    index = faiss.IndexFlatL2(hidden_dim)
    all_embeddings = []
    all_tokens = []
    for text in tqdm.tqdm(corpus):
        embeddings, tokens = get_embeddings_and_tokens(model, tokenizer, text, device)
        for i in range(len(tokens) - 1):
            all_embeddings.append(embeddings[i])
            all_tokens.append(tokens[i+1])
    embeddings_array = np.array(all_embeddings, dtype='float32')
    index.add(embeddings_array)
    return index, all_tokens, embeddings_array


def interactive_text_generator(model, tokenizer, index, tokens, device):
    print("Interactive Text Generator:")
    while True:
        input_text = input("Enter your text (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        tokens_to_generate = int(input("Tokens: "))
        k = int(input("K: "))
        generated_text = input_text
        for t in range(tokens_to_generate):
            next_token = predict_next_token(generated_text, model, tokenizer, index, tokens, k, device)
            print(next_token, end='')
            generated_text = generated_text + next_token
        print('------------------------')
        print(generated_text)

def get_embeddings_and_tokens(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)  # TODO
    with torch.no_grad():
        outputs = model(**inputs)
    assert len(inputs) == len(outputs), 'input output mismatch'
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # Remove batch dimension
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    assert len(embeddings) == len(tokens), 'embeddings/tokens mismatch'
    return embeddings, tokens


def predict_next_token(input_text, model, tokenizer, index, tokens, k, device):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    assert len(inputs) == len(outputs), 'input output mismatch'
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    last_embedding = embeddings[-1].reshape(1, -1).astype(np.float32)
    distances, indices = index.search(last_embedding, k)
    predicted_tokens = [tokens[i] for i in indices[0]]
    next_token = random.choice(predicted_tokens)
    next_token = next_token.replace(space_token, " ")
    return next_token


def main():
    logging.basicConfig(level=logging.INFO)

    torch.set_num_threads(1)

    device = torch.device(device_name)

    print(f'preparing {model_name} model...')
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    print(f'HF_TOKEN: {hf_token}')

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)

    if use_quantization:
        print('quantizing model')
        model = model.half()
    if device_name != 'cpu':
        print(f'moving to device: {device_name}')
        model.to(device)
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
        if len(corpus) >= 50:
            print('ending corpus early')
            break

    if os.path.exists('data.pkl'):
        print('loading index from pkl..')
        with open('data.pkl', 'rb') as f:
            (index, tokens, embeddings) = pickle.load(f)
    else:
        print('creating index...')
        t1 = time.time()
        index, tokens, embeddings = create_faiss_index(model, tokenizer, corpus, device)
        t2 = time.time()
        print(f'processing corpus took {(t2 - t1):.2f} seconds.')
        with open('data.pkl', 'wb') as f:
            pickle.dump((index, tokens, embeddings), f)

    # Start the interactive session
    interactive_text_generator(model, tokenizer, index, tokens, device)


if __name__ == '__main__':
    main()
