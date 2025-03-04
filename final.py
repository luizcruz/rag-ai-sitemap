import os
import json
import asyncio
import requests
import streamlit as st
import xml.etree.ElementTree as ET
import chromadb
import ollama
import re
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from sentence_transformers import SentenceTransformer

# 🔹 Configuração global
PASTA_JSON = "./data"
CHROMA_DB_PATH = "./chroma_db"
SITEMAP_URL = "https://www.lance.com.br/sitemap/news/today.xml"
MODEL_NAME = "deepseek-r1:8b"
MODEL_EMBEDDING = "all-MiniLM-L6-v2"
MAX_CRAWLERS = 5 

# 🔹 Conectar ao banco ChromaDB
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = db.get_or_create_collection(name="noticias")
modelo = SentenceTransformer(MODEL_EMBEDDING)

# 🔹 Função para extrair slug da URL
def extrair_slug(url):
    path = urlparse(url).path
    slug = path.rstrip('/').split('/')[-1]
    return os.path.splitext(slug)[0]

# 🔹 Função para extrair links do sitemap
def extrair_links_sitemap(url_sitemap):
    resposta = requests.get(url_sitemap)
    if resposta.status_code != 200:
        st.error(f"Erro ao acessar {url_sitemap}")
        return []
    
    root = ET.fromstring(resposta.text)
    urls = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return urls

# 🔹 Função para rodar um crawler em paralelo
async def rodar_crawler_para_link(link):
    prune_filter = PruningContentFilter(threshold=0.45, threshold_type="dynamic", min_word_threshold=5)
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)
    config = CrawlerRunConfig(markdown_generator=md_generator, cache_mode=CacheMode.BYPASS, excluded_tags=["nav", "footer", "header"], exclude_external_links=True)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=link, config=config)
        if result.success:
            mkResult = result.markdown_v2.fit_markdown
            match = re.search(r'^# (.+)', mkResult, re.MULTILINE)
            title = match.group(1) if match else "Sem título"

            json_data = {"title": title, "link": link, "content": mkResult}
            filename = os.path.join(PASTA_JSON, f"{extrair_slug(link)}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            return f"✅ Notícia salva: {title}"
        else:
            return f"❌ Erro ao extrair {link}: {result.error_message}"


async def rodar_crawler():
    
    os.makedirs(PASTA_JSON, exist_ok=True)
    links = extrair_links_sitemap(SITEMAP_URL)
    
    # Executar crawlers em paralelo com limite máximo de processos simultâneos
    resultados = []
    for i in range(0, len(links), MAX_CRAWLERS):
        batch = links[i:i+MAX_CRAWLERS]  # Divide os links em lotes de MAX_CRAWLERS
        resultados += await asyncio.gather(*(rodar_crawler_para_link(link) for link in batch))
    
    for resultado in resultados:
        st.write(resultado)

# 🔹 Função para carregar arquivos JSON e inserir no ChromaDB
def processar_arquivos_json():
    documentos = []
    for arquivo in os.listdir(PASTA_JSON):
        if arquivo.endswith(".json"):
            caminho_arquivo = os.path.join(PASTA_JSON, arquivo)
            with open(caminho_arquivo, "r", encoding="utf-8") as f:
                dados = json.load(f)

            slug = os.path.splitext(arquivo)[0]
            documentos.append({"id": slug, "title": dados.get("title", slug), "link": dados.get("link", ""), "content": dados.get("content", "")})

    for doc in documentos:
        embedding = modelo.encode(doc["content"]).tolist()
        collection.add(ids=[doc["id"]], embeddings=[embedding], metadatas=[{"title": doc["title"], "link": doc["link"], "content": doc["content"]}])

    st.success(f"{len(documentos)} documentos inseridos no ChromaDB!")


def apagar_arquivos_json():
    """Apaga todos os arquivos JSON dentro do diretório de dados."""
    if not os.path.exists(PASTA_JSON):
        st.warning("A pasta de dados não existe.")
        return
    
    arquivos = [f for f in os.listdir(PASTA_JSON) if f.endswith(".json")]
    
    if not arquivos:
        st.warning("Nenhum arquivo JSON encontrado para deletar.")
        return

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(PASTA_JSON, arquivo)
        try:
            os.remove(caminho_arquivo)
        except Exception as e:
            st.error(f"Erro ao deletar {arquivo}: {e}")

    st.success(f"🗑️ {len(arquivos)} arquivos JSON foram deletados com sucesso!")


# 🔹 Função para buscar contexto no ChromaDB
def buscar_contexto(consulta):
    consulta_embedding = modelo.encode(consulta).tolist()
    resultados = collection.query(query_embeddings=[consulta_embedding], n_results=3)
    
    contexto = ""
    for r in resultados["metadatas"][0]:
        contexto += f"\n🔹 {r['title']}\n📄 {r['content'][:500]}...\n🔗 {r['link']}\n\n"

    return contexto.strip() if contexto else "Nenhuma informação relevante encontrada."

# 🔹 Função para consultar o LLM com contexto
def perguntar_ao_llm(consulta):
    contexto = buscar_contexto(consulta)
    prompt = f"""Você é um assistente especializado em notícias. 
    Use as informações abaixo para responder à pergunta do usuário, incluindo os links das fontes:

    {contexto}

    Pergunta: {consulta}
    Resposta:"""

    resposta = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return resposta['message']['content']

# 🔹 Interface do Streamlit
st.set_page_config(page_title="Lance! RAG AI", page_icon="🤖")

# 🔹 Criar abas no Streamlit
aba = st.sidebar.radio("Navegação", ["📡 Crawler", "📥 Vetorização", "🗑️ Apagar Local",  "💬 Chat"])

if aba == "📡 Crawler":
    st.title("📡 Crawler - Extração de Notícias")
    st.write("Clique no botão abaixo para iniciar a coleta de notícias do site Lance!")

    if st.button("Iniciar Crawler"):
        st.warning("O processo pode levar alguns minutos...")
        asyncio.run(rodar_crawler())

elif aba == "📥 Vetorização":
    st.title("📥 Vetorização - Inserir Notícias no ChromaDB")
    st.write("Processar os arquivos JSON e armazenar as notícias no banco vetorial.")

    if st.button("Processar Arquivos"):
        processar_arquivos_json()

elif aba == "🗑️ Apagar Local":
    st.title("🗑️ Apagar Arquivos JSON")
    st.write("Clique no botão abaixo para apagar todos os arquivos JSON extraídos.")

    if st.button("Apagar Arquivos JSON"):
        apagar_arquivos_json()

elif aba == "💬 Chat":
    st.title("🤖 LANCE AI")
    st.write("Digite uma pergunta e obtenha respostas baseadas em notícias armazenadas no banco vetorial.")

    consulta_usuario = st.text_input("Pergunte:", placeholder="Quem joga hoje?")
    
    if st.button("Enviar"):
        if consulta_usuario:
            with st.spinner("🔍 Buscando resposta..."):
                resposta = perguntar_ao_llm(consulta_usuario)
            
            st.subheader("🧠 Resposta do LLM:")
            st.write(resposta)
        else:
            st.warning("Por favor, digite uma pergunta antes de continuar.")

# 🔹 Rodapé
st.markdown("---")
st.markdown("⚡ Desenvolvido com **Ollama + Crawl4AI + ChromaDB + Streamlit + DeepSeek-R1:8B**")
