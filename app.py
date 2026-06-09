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
# CORREÇÃO DA LINHA 67: String fechada corretamente com aspas e parêntese
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Caso o histórico esteja vazio, adiciona a mensagem inicial de acolhimento
if len(msgs.messages) == 0:
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")

# Envelopando a nossa chain para gerenciar automaticamente a persistência do histórico
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs, 
    input_messages_key="input",
    history_messages_key="history"
)

# 7. Configuração da Barra Lateral
st.sidebar.title("Configurações")
st.sidebar.write("Modelo: Zephyr-7B-Chat")
st.sidebar.write("Framework: LangChain LCEL")

limpar_conversa = st.sidebar.button("🧹 Limpar Histórico")

if limpar_conversa:
    msgs.clear()
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")
    st.rerun()

# 8. Renderização da Interface do Chat
for msg in msgs.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 9. Fluxo de Entrada do Usuário e Resposta da IA em Tempo Real
if user_input := st.chat_input("Converse com o Companion..."):
    # Exibe imediatamente a mensagem digitada pelo usuário
    with st.chat_message("user"):
        st.write(user_input)
    
    # Chama a execução da corrente persistente do LangChain
    with st.chat_message("assistant"):
        with st.spinner("Pensando com cuidado..."):
            config = {"configurable": {"session_id": "companion_session"}}
            response = chain_with_history.invoke({"input": user_input}, config=config)
            answer_text = response.content
            st.write(answer_text)
