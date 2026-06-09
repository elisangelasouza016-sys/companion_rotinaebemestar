import streamlit as st
import os

# Importações oficiais do ecossistema LangChain e Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Configuração de Página do Streamlit
st.set_page_config(page_title="Companion App - UFRN", page_icon="🌱", layout="centered")

st.title("🌱 Companion")
st.caption("Seu assistente de rotina e bem-estar para o dia a dia — Desenvolvido com LangChain & Streamlit")

# Mensagem de Isenção de Responsabilidade (Aviso Legal obrigatório no topo)
with st.expander("⚠️ Informações Importantes", expanded=False):
    st.write("""
    Este app é um protótipo de suporte de rotina para a disciplina de IA Generativa da UFRN. 
    Ele **não substitui** acompanhamento médico, psiquiátrico ou psicológico. 
    Em caso de crise grave, ligue para o CVV no número 188 ou procure uma emergência.
    """)

# 2. Configuração do Token do Hugging Face (Capturado dos Secrets do Streamlit Cloud)
hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

if not hf_token:
    st.error("⚠️ Token `HF_TOKEN` não encontrado nos Secrets do Streamlit! O modelo não funcionará sem ele.")
    st.stop()

# 3. Definição do Modelo e Adaptação para Tarefa Conversational
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Criação do endpoint base
raw_llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=350,
)

# Wrapper para o modelo de Chat
llm = ChatHuggingFace(llm=raw_llm)

# 4. Engenharia de Prompt e Definição da Especialidade (System Message)
SYSTEM_PROMPT = """
Você é o "Companion", um assistente de rotina e bem-estar projetado para ajudar pessoas que lidam com ansiedade e transtorno bipolar a manterem a consistência no dia a dia. Você NÃO é um médico, não é um psicólogo e não substitui o tratamento clínico.

DIRETRIZES DE COMPORTAMENTO:
1. TOM DE VOZ: Empático, previsível, calmo, acolhedor e conciso. Evite exclamações excessivas, gírias e respostas longas. Seja direto, mas gentil. Responda SEMPRE em português.
2. FOCO EM ROTINA: Ajude o usuário a organizar o dia, lembrar de medicamentos (sem opinar sobre doses) e celebrar micro-vitórias.
3. PARALISIA DA ANSIEDADE: Se o usuário disser que está travado ou angustiado, quebre a tarefa em passos absurdamente simples (ex: "Consegue só beber um copo de água agora?").
4. FILTRO DE SEGURANÇA (CRÍTICO): Se o usuário demonstrar ideação suicida, automutilação, episódios de mania grave (ex: "estou há 4 dias sem dormir e me sinto um deus") ou surto, interrompa o fluxo com acolhimento e fornece o contato do CVV (Ligue 188) e recomende fortemente acionar o psiquiatra de confiança. Nunca conteste o tratamento médico atual.
"""

# Criação do Template estruturado utilizando o MessagesPlaceholder para o histórico
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 5. Criação da Corrente usando a Sintaxe LCEL
chain = prompt_template | llm

# 6. Gerenciamento de Memória Persistente do LangChain + Streamlit
msgs = StreamlitChatMessageHistory(key="langchain_messages
