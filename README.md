# RAG Demos
These are demo implementations of selected use cases and wil be continuously extended. You can find accompanying descriptions in my blog at https://operational-ai-blog.de . The first use case implemented deals with gaps with regulations like GDPR with own policy documents.

## Installation

This repository involves ingesting documents into vector databases. Especially for PDF files I found the pipeline from PDF to MS Word to Markdown most reliable. Therefore that part only works on MS Windows. Once the embeddings are created the solution can be migrated to Linux and continued there.

The following additional software is required:

1. Docker: on Windows you can install Docker Desktop from here: https://www.docker.com/products/docker-desktop/ . Docker is used to install a Postgres instance and its vector extension as well as an accompanying pgadmin instance in a separate container. Another container installs Ollama for locally installed LLM and Embedding models if needed.
2. GIT: I recommend GitBash for Windows. 
3. Python: I used pyenv with a python version of 3.11.9 . For the installation of pyenv and hte required packages please see the section below.

### Installing the repo

Use the following command to checkout this repository out:

```
cd YOUR_GIT_DIRECTORY
git clone https://github.com/rgraefe/rag-demo.git
```

### Installing pyenv

Clone and install pyenv as described in https://github.com/pyenv-win/pyenv-win .
Then install the used python version, create a virtual environment and install the required packages.

```
pyenv install 3.11.9
pyenv global 3.11.9
cd YOUR_GIT_DIRECTORY\rag-demo
pyenv local 3.11.9
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -c "import pypandoc; pypandoc.download_pandoc()"
```
The last command is necessary because the pypandoc python library also needs a windows component to work.

### Starting the docker containers

Before starting docker you need to add one file in directory pgadmin:

```
cd pgadmin
```

Create a file named pgadmin.env with your favourite editor. Enter 2 values.

```
PGADMIN_DEFAULT_EMAIL=<pgadmin login email>
PGADMIN_DEFAULT_PASSWORD=<pgadmin password>
```

Chose a login email and password that you can later use to login to the pgadmin console to access your postgres database for debugging reasons.

The email you chose will determine some other docker values. 
First open the `docker-compose.yaml`. Look for entries of the form `.../ralf_mrghome.de/...` . This is my email address with the '@' character replaced by '\_' . Please insert the same email address you used for PGADMIN_DEFAULT_EMAIL, replace '@' with '\_' and insert in the respective positions. There is a second location where this needs to be changed: `pgadmin/config/server.json`.

After that you can start the docker containers:
Cd into the repository root and there type `docker compose up -d`. You will see some status messages fetching the respective docker containers. After successfull start you should see something like this:

```
docker ps
CONTAINER ID   IMAGE               COMMAND                  CREATED      STATUS      PORTS                                     NAMES
7c56f4733b75   rag-demo-postgres   "docker-entrypoint.s…"   2 days ago   Up 2 days   127.0.0.1:5433->5432/tcp                  rag-demo-postgres-1
bd35f9564121   rag-demo-pgadmin    "/bin/sh -c ' cp -f …"   2 days ago   Up 2 days   0.0.0.0:8484->80/tcp, [::]:8484->80/tcp   rag-demo-pgadmin-1
c55b29c7192b   ollama/ollama       "/bin/ollama serve"      2 days ago   Up 2 days   127.0.0.1:11434->11434/tcp                ollama
```

After spawning the docker container for ollama add the actual models that are served by commands like (on windows drop sudo):

```
sudo docker exec -it ollama ollama pull llama3.2
sudo docker exec -it ollama ollama pull nomic-embed-text
sudo docker exec -it ollama ollama pull mxbai-embed-large
sudo docker exec -it ollama ollama pull snowflake-arctic-embed2
```

### Setting environment variables

Create a file `.env`  in the root directory. 
Add a the line `OPENAI_API_KEY='sk-proj--########################################'`
Replace sk-proj... with your own OpenAI Api key.
If you don't want to use OpenAI you can try llama3 installed in the Ollama docker container.

## Usage

The directory `notebooks` contains several Jupyter notebooks demonstrating different aspects of the use case implementation. Notebook `create_new_vectorindex.ipynb` demonstrate parsing and storing PDF files for the Regulatory Gap Analysis use case. Notebook `parent_retriever.ipynb` demonstrates retrieving parent/child related documents based on the parsed PDF files.
