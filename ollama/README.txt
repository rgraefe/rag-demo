After spawning the docker container for ollama add the actual models that are served by commands like:

sudo docker exec -it ollama ollama pull llama3.2
sudo docker exec -it ollama ollama pull nomic-embed-text
sudo docker exec -it ollama ollama pull mxbai-embed-large
sudo docker exec -it ollama ollama pull snowflake-arctic-embed2