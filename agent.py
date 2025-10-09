from langchain.memory import ConversationBufferMemory
# Você pode trocar por LangGraph se preferir um grafo de decisão
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_debug
# Construção de prompts
from langchain.schema import HumanMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Parsers de saída
from langchain.schema.output_parser import StrOutputParser

# Execução de fluxos (Runnables)
from langchain_core.runnables import RunnableLambda

# Criação e execução de agentes
from langchain.agents import (
    Tool,
    AgentExecutor,
    create_tool_calling_agent,
    create_react_agent
)
# Ferramentas customizadas para agentes
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain import hub
import os

# Tipos de memória utilizados em agentes
from langchain.memory import ConversationBufferMemory

# Componentes de RAG (Retrieval-Augmented Generation)
from langchain_chroma import Chroma  # Armazenamento vetorial
from langchain_openai.embeddings import OpenAIEmbeddings  # Embeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings  # Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Separador de texto
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Leitura de documentos PDF
# =======================
# LangGraph
# =======================

# Criação de agentes com LangGraph
from langgraph.prebuilt import create_react_agent as create_react_agent_graph

# Sistema de checkpoint em memória
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent import RunnableAgentType

# Definição e execução de grafos
from langgraph.graph import StateGraph, END

OUTPUT_DOCUMENTS_DIR:str = 'data/'

class EDAAgent:
    
    vectorstore = None
    OUTPUT_CSV_DIR = ''
    embeddings = None
    
    def __init__(self, df):
        self.df = df
        self.memory = ConversationBufferMemory(k=100)
        
    def cria_banco_de_dados_vetorial(file_path:str) -> None:
        try:
            # Carrega os documentos do diretório especificado
            documents = CSVLoader(file_path).load()

            # Usando embeddings do OpenAI
            embeddings = GoogleGenerativeAIEmbeddings(model="models/")

            # Cria um banco de dados vetorial usando Chroma
            split_documents = RecursiveCharacterTextSplitter(chunk_size=480, chunk_overlap=100).split_documents(documents)

            # Cria o banco de dados vetorial
            vectorstore = Chroma.from_documents(split_documents, embeddings, persist_directory=f'{OUTPUT_DOCUMENTS_DIR}vectorstore')
            print("Banco de dados vetorial criado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar documentos: {e}")

    def carrega_banco_de_dados_vetorial(path_documentos:str) -> Chroma:
        try:
            # Carrega o banco de dados vetorial existente
            #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma(persist_directory=path_documentos, embedding_function=embeddings)
            return vectorstore
        except Exception as e:
            print(f"Erro ao carregar o banco de dados vetorial: {e}")
            return None
        
    def busca_na_base_de_documentos(pergunta:str) -> str:
        """Você é um especialista em Análise Exploratória de Dados no contexto de fraude de cartão de crédito. 
        Use esta ferramenta para responder perguntas técnicas, exibir gráficos quando aplicáveis.
        """
        vectorstore = EDAAgent.carrega_banco_de_dados_vetorial(f'{OUTPUT_DOCUMENTS_DIR}vectorstore')
        contexto = None
        if vectorstore:
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs])
        return contexto
        
    vectorstore = carrega_banco_de_dados_vetorial(f'{OUTPUT_DOCUMENTS_DIR}vectorstore')
    docs = None

    if vectorstore:
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke("Data H")
        print(docs)
    else:
        print("Não foi possível carregar o banco de dados vetorial.")

    def agente_langchain(llm:BaseChatModel) -> dict:
        ferramentas = [PythonAstREPLTool()]

        prompt = hub.pull("hwchase17/react", api_key=os.environ['LANGSMITH_API_KEY'])
        print('\n','-'*40,'\n',prompt.template, '\n','-'*40, '\n')

        # To query a single csv file
        #executor_do_agente = create_csv_agent(
        #    llm,
        #    "data/eda.csv",
        #    verbose=True,
        #    agent_type=RunnableAgentType.from_llm_and_tools(llm),
        #)
        agente = create_react_agent(llm, ferramentas, prompt)
        executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas, handle_parsing_errors=True)
        return executor_do_agente
    
    def query(self, question):
        llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-1.5-flash-latest') 
        #contexto = EDAAgent.busca_na_base_de_documentos(question) or ''
        executor_do_agente = EDAAgent.agente_langchain(llm)
        resposta = executor_do_agente.invoke({"input": question, "context": contexto})
        return resposta.get("output", "Não encontrei a resposta")