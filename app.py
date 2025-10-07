import streamlit as st
import pandas as pd
from eda_utils import run_quick_eda, plot_eda_figures
from agent import EDAAgent
import io, requests, zipfile

# Carregamento de dados
def load_data(uploaded_file, url, use_demo):
    if use_demo:
        return pd.read_csv('data/demo_creditcard_fraud.csv')
    elif uploaded_file is not None:
        if uploaded_file.name.endswith('.zip'):
            z = zipfile.ZipFile(uploaded_file)
            csv_names = [n for n in z.namelist() if n.endswith('.csv')]
            if not csv_names: st.error('ZIP não contém CSV'); return None
            return pd.read_csv(z.open(csv_names[0]))
        else:
            return pd.read_csv(uploaded_file)
    elif url:
        content = requests.get(url).content
        return pd.read_csv(io.BytesIO(content))
    else:
        st.info('Escolha alguma forma de carregar os dados.')
        return None

st.set_page_config(page_title="EDA Agente IA", layout="wide")
st.title("Sistema de Análise Exploratória de Dados com IA 🤖📊")

# Sidebar para upload/url/demo
with st.sidebar:
    st.header('Carregue seus dados')
    upload = st.file_uploader("CSV/ZIP", type=['csv','zip'])
    url = st.text_input('Ou informe uma URL:')
    use_demo = st.button("Usar Dataset Demo")
    st.markdown("---")

# Carregamento de dados
if use_demo or upload or url:
    df = load_data(upload, url, use_demo)
else:
    df = None

if df is not None:
    st.subheader("Amostra dos dados")
    st.dataframe(df.head())
    st.write(f"Tamanho: {df.shape[0]} linhas, {df.shape[1]} colunas")
    st.write("Colunas:", list(df.columns))

    # Seção chat com agente
    st.subheader("Pergunte ao agente sobre os dados 📢")
    if 'eda_agent' not in st.session_state:
        st.session_state.eda_agent = EDAAgent(df)
        st.session_state.chat_history = []
    agent = st.session_state.eda_agent

    chat_input = st.text_input("Digite sua pergunta:")
    if st.button("Enviar"):
        resposta, figs = agent.query(chat_input)
        st.session_state.chat_history.append(("Usuário", chat_input))
        st.session_state.chat_history.append(("Agente", resposta))
        st.write(resposta)
        for fig in figs:
            st.pyplot(fig)
    if st.button("Limpar Histórico"): st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")

    st.subheader("Análises rápidas")
    for op in ["Descrição", "Correlação", "Outliers", "Distribuição", "Classes"]:
        if st.button(op):
            resp, figs = run_quick_eda(df, op)
            st.write(resp)
            for fig in figs:
                st.pyplot(fig)
else:
    st.write("Carregue um dataset para iniciar sua análise.")

st.markdown("---")
st.markdown('Feito com Streamlit, LangChain e Python | Publicável no Render.com')
