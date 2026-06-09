import streamlit as st
import os

# Importações obrigatórias do ecossistema LangChain e Hugging Face
from langchain_huggingface import HuggingFaceEndpoint
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

# 3. Definição do Modelo Especializado (Homologando o Qwen 2.5)
# Parâmetros de exploração exigidos pela atividade: Temperatura estável e limite de novos tokens
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=hf_token,
    temperature=0.5,           # Parâmetro explorado: equilíbrio entre precisão e empatia
    max_new_tokens=350,        # Parâmetro explorado: evita respostas excessivamente longas
    task="text-generation"
)

# 4. Engenharia de Prompt e Definição da Especialidade (System Message)
SYSTEM_PROMPT = """
Você é o "Companion", um assistente de rotina e bem-estar projetado para ajudar pessoas que lidam com ansiedade e transtorno bipolar a manterem a consistência no dia a dia. Você NÃO é um médico, não é um psicólogo e não substitui o tratamento clínico.

DIRETRIZES DE COMPORTAMENTO:
1. TOM DE VOZ: Empático, previsível, calmo, acolhedor e conciso. Evite exclamações excessivas, gírias e respostas longas. Seja direto, mas gentil. Responda SEMPRE em português.
2. FOCO EM ROTINA: Ajude o usuário a organizar o dia, lembrar de medicamentos (sem opinar sobre doses) e celebrar micro-vitórias.
3. PARALISIA DA ANSIEDADE: Se o usuário disser que está travado ou angustiado, quebre a tarefa em passos absurdamente simples (ex: "Consegue só beber um copo de água agora?").
4. FILTRO DE SEGURANÇA (CRÍTICO): Se o usuário demonstrar ideação suicida, automutilação, episódios de mania grave (ex: "estou há 4 dias sem dormir e me sinto um deus") ou surto, interrompa o fluxo com acolhimento e forneça o contato do CVV (Ligue 188) e recomende fortemente acionar o psiquiatra de confiança. Nunca conteste o tratamento médico atual.
"""

# Criação do Template estruturado utilizando o MessagesPlaceholder para o histórico
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"), # Espaço reservado dinamicamente para o histórico
    ("human", "{input}")
])

# 5. Criação da Corrente usando a Sintaxe LCEL (LangChain Expression Language)
# O operador pipe (|) encadeia o prompt de entrada diretamente ao modelo de linguagem
chain = prompt_template | llm

# 6. Gerenciamento de Memória Persistente do LangChain + Streamlit
# O StreamlitChatMessageHistory cuida de salvar as mensagens na sessão de forma automatizada
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Caso o histórico esteja vazio, adiciona a mensagem inicial de acolhimento do assistente
if len(msgs.messages) == 0:
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")

# Envelopando a nossa chain para gerenciar automaticamente a persistência do histórico
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs, # Retorna a instância de histórico do Streamlit
    input_messages_key="input",
    history_messages_key="history"
)

# 7. Configuração da Barra Lateral (Requisito: Botão de limpar histórico)
st.sidebar.title("Configurações do App")
st.sidebar.markdown("""
**Modelo:** `Qwen2.5-72B-Instruct`
**Framework:** LangChain LCEL
""")

if st.sidebar.button("🧹 Limpar Histórico de Conversa"):
    msgs.clear()
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")
    st.rerun()

# 8. Renderização da Interface do Chat
# Exibe todas as mensagens salvas na memória (pulando a system message interna do LangChain)
for msg in msgs.messages:
    # Mapeia os papéis do LangChain para os ícones visuais do Streamlit
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
            # O RunnableWithMessageHistory intercepta a chamada, injeta o histórico e executa a chain
            config = {"configurable": {"session_id": "companion_session"}}
            response = chain_with_history.invoke({"input": user_input}, config=config)
            
            # Renderiza a resposta final do modelo na tela
            st.write(response)
