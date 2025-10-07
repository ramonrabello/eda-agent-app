from langchain.memory import ConversationBufferMemory
# Você pode trocar por LangGraph se preferir um grafo de decisão

class EDAAgent:
    def __init__(self, df):
        self.df = df
        self.memory = ConversationBufferMemory(k=100)
    def query(self, question):
        # Aqui, use seu modelo LLM favorito, template ou lógica simples. Exemplo básico:
        if "correlação" in question.lower():
            corr = self.df.corr()
            return (f"As principais correlações são:
{corr.to_string()}", [])
        elif "fraude" in question.lower():
            n_fraudes = (self.df['Class'] == 1).sum()
            rate = n_fraudes / len(self.df)
            return (f"Número de fraudes detectadas: {n_fraudes} ({rate:.3%})", [])
        else:
            return ("Não entendi a pergunta. Reformule ou peça uma análise específica.", [])
