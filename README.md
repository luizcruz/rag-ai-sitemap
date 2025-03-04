# RAG AI usando sitemap

O Lance! RAG AI é um sistema completo de extração de notícias, processamento semântico e geração de respostas inteligentes baseado em banco vetorial. Ele combina Crawler, ChromaDB e LLMs locais via Ollama para oferecer uma experiência de busca semântica avançada.

O projeto é dividido em três etapas principais, acessíveis através de um dashboard interativo no Streamlit

Dentro do final.py há variáveis de controle a saber: 

PASTA_JSON  - Pasta que o crawler salva os conteúdos após raspagem
CHROMA_DB_PATH - Diretório padrão do ChromaDB
SITEMAP_URL - Sitemap para obtenção dos links
MODEL_NAME - Modelo de LLM escolhido
MODEL_EMBEDDING - Modelo de processo de vetorização (embedding)
MAX_CRAWLERS - Máximo de crawlers simultâneos (cuidado!)


## Requisitos


Verifique se já não possui Python com o comando: 

```
python3 --version
```

Caso negativo, instale com o comando: 

```
sudo apt-get install python3
```

Após ter o Python3 disponível, instale as dependências: 

```
pip install streamlit ollama chromadb sentence-transformers crawlforai requests lxml xmltodict
```

Para instalar o Ollama e o modelo DeepSeek R1 8b (no WSL Ubuntu)

```
curl https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:8b
```

Para testar local, você pode rodá-lo com o comando:

```
ollama run deepseek-r1:8b
```


## Rodando o Streamlit

Para rodar o projeto todo, rode o comando: 
```
streamlit run final.py
```

Abra no browser o endereço do servidor e use os recursos. 