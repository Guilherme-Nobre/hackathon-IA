import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# === 0. Corrigir conflito de bibliotecas OpenMP ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 1. Imports LangChain ===
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader

# === 2. Carregar variáveis de ambiente ===
load_dotenv()

# === 3. NOVO: Função cacheada para processar VÁRIOS PDFs ===
@st.cache_resource
def create_rag_chain_from_pdfs(pdf_bytes_tuple): # MUDANÇA: Recebe uma tupla de bytes
    """
    Processa VÁRIOS PDFs (em bytes), cria um RAG chain e o retorna.
    Fica em cache para evitar reprocessamento.
    """
    
    all_texts = [] # Lista para acumular textos de TODOS os PDFs

    # Definir o splitter uma vez, fora do loop
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100
    )

    # MUDANÇA: Iterar sobre os bytes de cada PDF enviado
    for pdf_bytes in pdf_bytes_tuple:
        # Usar um arquivo temporário para o PyMuPDFLoader (para cada PDF)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            file_path = temp_file.name

        try:
            # === 4. Leitura e divisão dos textos (por arquivo) ===
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            texts = splitter.split_documents(docs)
            all_texts.extend(texts) # Adicionar os textos deste PDF à lista total

        finally:
            # Limpar o arquivo temporário
            os.unlink(file_path)

    # === 5. Verificação (Após processar TODOS os arquivos) ===
    if not all_texts:
        # Retorna None se nenhum texto foi extraído de NENHUM PDF
        return None

    # === 6. Embeddings e vetorstore (Feito UMA VEZ com todos os textos) ===
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_texts, embeddings)

    # === 7. Criação do Retriever e Chain RAG ===
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return rag_chain


# === 8. Interface Streamlit ===
st.set_page_config(page_title="Chat com Múltiplos PDFs", layout="wide") # Título atualizado
st.title("📄 Chat com Múltiplos PDFs usando RAG e IA Generativa")

# MUDANÇA: 'accept_multiple_files=True' e variável no plural 'pdf_files'
pdf_files = st.file_uploader("Envie um ou mais PDFs", type=["pdf"], accept_multiple_files=True)

if pdf_files: # MUDANÇA: 'pdf_files' agora é uma lista
    
    # MUDANÇA: Ler os bytes de CADA arquivo e criar uma tupla (para o cache)
    pdf_bytes_list = [file.read() for file in pdf_files]
    pdf_bytes_tuple = tuple(pdf_bytes_list)
    
    # MUDANÇA: Chamar a nova função que aceita múltiplos arquivos
    rag_chain = create_rag_chain_from_pdfs(pdf_bytes_tuple)

    # === 9. Lidar com o caso de PDFs sem texto ===
    if rag_chain is None:
        st.error("Erro: Não foi possível extrair texto dos PDFs. Verifique se os arquivos não são apenas imagens escaneadas ou se não estão vazios.")
    else:
        # Mensagem de sucesso atualizada
        st.success(f"✅ {len(pdf_files)} PDF(s) processados com sucesso! Agora você pode fazer perguntas sobre o conteúdo combinado.")

        # === 10. Interface de perguntas (Permanece igual) ===
        user_question = st.text_input("❓ Pergunte algo sobre os documentos:")

        if user_question:
            with st.spinner("🔍 Consultando o conteúdo dos documentos..."):
                resposta = rag_chain.invoke({"query": user_question})

            st.markdown("### 🧠 Resposta:")
            st.write(resposta["result"])

            # Exibir fontes (opcional)
            with st.expander("📚 Fontes consultadas"):
                for i, doc in enumerate(resposta["source_documents"]):
                    st.markdown(f"**Trecho {i+1}:**")
                    st.write(doc.page_content)