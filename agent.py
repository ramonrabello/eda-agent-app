# agent.py

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, ToolNode
from langchain.agents import tool
from langchain.agents.react.agent import create_react_agent

# Definir ferramentas para o agente (Exemplo: análise descritiva)
@tool
def describe_data(df):
    """Retorna estatísticas descritivas para o dataframe."""
    return str(df.describe())

# Defina seu modelo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Crie o agente ReAct via LangGraph
tools = [describe_data]
react_agent = create_react_agent(llm, tools)

# Exemplo de orquestração com LangGraph
def run_agent(question, df):
    # Crie o grafo de estados
    graph = StateGraph()
    # Adicione o nó do agente (ele pode acionar ferramentas)
    agent_node = ToolNode(
        agent=react_agent,
        tools={'describe_data': lambda: describe_data(df)}
    )
    graph.add_node('react_agent', agent_node)
    graph.set_entry_point('react_agent')
    # Execute o grafo passando a pergunta
    result = graph.run({'input': question})
    return result['output']
