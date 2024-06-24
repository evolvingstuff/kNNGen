## kNNGen

Experimenting with some of the ideas in this paper:

[Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172)

and later might incorporate ideas from this paper as well:

[Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)

---

### Setup

Requires Docker.

Install and run Milvus, as explained [here](https://milvus.io/docs/install_standalone-docker.md):
```
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start
```
Optionally run `milvus_test.py` to see if that worked.

Create a `.env` file, and inside of it add your HuggingFace API token, like so:
```
HF_TOKEN=your_hugging_face_api_token_here
```
*Or* add the equivalent to your system's environment variables.