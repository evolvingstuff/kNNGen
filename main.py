import pickle
import random
import time
from dotenv import load_dotenv
import tqdm
import torch
import numpy as np
import logging
import os


# TODO move to config file
K = 1
threads = 1  # None
device_name = 'mps'  # 'mps'
limit_corpus_size = None  # 15_000
# model_name = 'gpt2'
# space_token = "Ġ"
# model_name = 'mistralai/Mistral-7B-v0.1'
space_token = '▁'
model_name = 'google/gemma-2B'

dataset_config = {
    "path": "wikitext",
    "name": "wikitext-2-raw-v1"
}
max_length = 512
use_quantization = False


# avoid potential OpenMP version conflicts between huggingface and faiss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if threads is not None:
    torch.set_num_threads(threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
import faiss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel)


def create_faiss_index(model, tokenizer, corpus, device):
    hidden_dim = model.config.hidden_size  # model.config.n_embd

    t1 = time.time()
    safe_model_name = model_name.replace('/', '.')
    data_file = f'{safe_model_name}.data.pkl'
    if os.path.exists(data_file):
        print("Loading embeddings from google.gemma-2B.data.pkl")
        with open(data_file, "rb") as f:
            all_tokens, all_embeddings = pickle.load(f)
        t2 = time.time()
        print(f'done, took {(t2-t1):.2f}s')
    else:
        print('calculating embeddings...')
        all_embeddings = []
        all_tokens = []
        t1 = time.time()
        for text in tqdm.tqdm(corpus):
            embeddings, tokens = get_embeddings_and_tokens(model, tokenizer, text, device)
            for i in range(len(tokens) - 1):
                all_embeddings.append(embeddings[i])
                all_tokens.append(tokens[i+1])
        t2 = time.time()
        print(f'done calculating tokens and embeddings arrays, took {(t2-t1):.2f}s')
        print('saving embeddings...')
        with open(data_file, 'wb') as f:
            pickle.dump((all_tokens, all_embeddings), f)
        t2 = time.time()
        print(f'done pickling, took {(t2 - t1):.2f}s')
    # TODO: break this out into separate function?
    print('adding embeddings array to faiss index...')
    t1 = time.time()
    embeddings_array = np.array(all_embeddings, dtype='float32')


    # index = faiss.IndexFlatL2(hidden_dim)
    # index.add(embeddings_array)

    nlist = 100  # Number of clusters
    m = 8  # Number of sub-vector quantizations
    quantizer = faiss.IndexFlatL2(hidden_dim)  # This remains the same
    index = faiss.IndexIVFPQ(quantizer, hidden_dim, nlist, m, 8)

    print('cp1')

    try:
        # You need to train the index
        subset = embeddings_array[:5000]
        index.train(subset)  # training_data is a subset of your dataset
    except Exception as e:
        print('uh oh')
        print(e)
        raise e

    print('cp2')

    # After training, you can add your vectors
    index.add(embeddings_array)

    t2 = time.time()
    print(f'done, took {(t2-t1):.2f}s')
    return index, all_tokens, embeddings_array


def interactive_text_generator(model, tokenizer, index, tokens, device):
    print("Interactive Text Generator:")
    # torch.mpu.empty_cache()  # TODO vary per architecture
    while True:
        input_text = input("Enter your text (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        tokens_to_generate = int(input("Tokens: "))
        k = int(input("K: "))
        generated_text = input_text
        warmup = None
        for t in range(tokens_to_generate):
            next_token, llm_time, faiss_time = predict_next_token(generated_text, model, tokenizer, index, tokens, k, device, space_token)
            print(next_token, end='')
            generated_text = generated_text + next_token
            if warmup is None:
                warmup = llm_time
        print('')
        print('------------------------')
        print(f'warmup: {warmup:.2f}s')
        print(generated_text)


def get_embeddings_and_tokens(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    assert len(inputs) == len(outputs), 'input output mismatch'
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # Remove batch dimension
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu().numpy())
    assert len(embeddings) == len(tokens), 'embeddings/tokens mismatch'
    return embeddings, tokens


def predict_next_token(input_text, model, tokenizer, index, tokens, k, device, space_token):
    t1 = time.time()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    assert len(inputs) == len(outputs), 'input output mismatch'
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    last_embedding = embeddings[-1].reshape(1, -1).astype(np.float32)
    t2 = time.time()
    llm_time = t2 - t1
    t1 = time.time()
    distances, indices = index.search(last_embedding, k)
    t2 = time.time()
    faiss_time = t2 - t1
    predicted_tokens = [tokens[i] for i in indices[0]]
    next_token = random.choice(predicted_tokens)
    next_token = next_token.replace(space_token, " ")
    return next_token, llm_time, faiss_time


def main():
    logging.basicConfig(level=logging.INFO)

    device = torch.device(device_name)

    print(f'preparing {model_name} model...')
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    print(f'HF_TOKEN: {hf_token}')

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    safe_model_name = model_name.replace('/', '.')
    if use_quantization:
        model_file = f'{safe_model_name}.half.bin'
    else:
        model_file = f'{safe_model_name}.bin'

    if os.path.exists(model_file):
        print('loading model from saved file...')
        t1 = time.time()
        model = torch.load(model_file)
        t2 = time.time()
        print(f'done loading model, took {(t2-t1):.2f}s')
    else:
        print('loading model...')
        t1 = time.time()
        model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)
        if use_quantization:
            print('quantizing model')
            model = model.half()
        t2 = time.time()
        print(f'done loading model, took {(t2-t1):.2f}s')
        print('saving model...')
        t1 = time.time()
        torch.save(model, model_file)
        t2 = time.time()
        print(f'done saving model, took {(t2-t1):.2f}s')

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
        if limit_corpus_size is not None and len(corpus) >= limit_corpus_size:
            print(f'ending corpus early at {limit_corpus_size} examples')
            break

    print('creating index...')
    t1 = time.time()
    index, tokens, embeddings = create_faiss_index(model, tokenizer, corpus, device)
    t2 = time.time()
    print(f'processing corpus took {(t2 - t1):.2f} seconds.')

    # Start the interactive session
    interactive_text_generator(model, tokenizer, index, tokens, device)


if __name__ == '__main__':
    main()
