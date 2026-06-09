import streamlit as st
import os

# Importações oficiais do ecossistema LangChain e Hugging Face
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Configuração de Página do Streamlit [cite: 34, 35]
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

# 3. Definição do Modelo e Adaptação para Tarefa Conversational [cite: 10, 22]
# Zephyr é totalmente compatível com a classe ChatHuggingFace e aceita o formato 'conversational' do Hub sem dar BadRequest
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Criação do endpoint base
raw_llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    huggingfacehub_api_token=hf_token,
    temperature=0.5,           # Parâmetro explorado: equilíbrio entre precisão e empatia [cite: 23, 26]
    max_new_tokens=350,        # Parâmetro explorado: evita respostas longas [cite: 23, 27]
)

# Wrapper que resolve de vez o erro de validação de modelo de Chat do provedor
llm = ChatHuggingFace(llm=raw_llm)

# 4. Engenharia de Prompt e Definição da Especialidade (System Message) [cite: 15, 16, 19]
SYSTEM_PROMPT = """
Você é o "Companion", um assistente de rotina e bem-estar projetado para ajudar pessoas que lidam com ansiedade e transtorno bipolar a manterem a consistência no dia a dia. Você NÃO é um médico, não é um psicólogo e não substitui o tratamento clínico.

DIRETRIZES DE COMPORTAMENTO:
1. TOM DE VOZ: Empático, previsível, calmo, acolhedor e conciso. Evite exclamações excessivas, gírias e respostas longas. Seja direto, mas gentil. Responda SEMPRE em português.
2. FOCO EM ROTINA: Ajude o usuário a organizar o dia, lembrar de medicamentos (sem opinar sobre doses) e celebrar micro-vitórias.
3. PARALISIA DA ANSIEDADE: Se o usuário disser que está travado ou angustiado, quebre a tarefa em passos absurdamente simples (ex: "Consegue só beber um copo de água agora?").
4. FILTRO DE SEGURANÇA (CRÍTICO): Se o usuário demonstrar ideação suicida, automutilação, episódios de mania grave (ex: "estou há 4 dias sem dormir e me sinto um deus") ou surto, interrompa o fluxo com acolhimento e fornece o contato do CVV (Ligue 188) e recomende fortemente acionar o psiquiatra de confiança. Nunca conteste o tratamento médico atual.
"""

# Criação do Template estruturado utilizando o MessagesPlaceholder para o histórico [cite: 29, 31]
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"), # Espaço reservado dinamicamente para o histórico [cite: 31]
    ("human", "{input}")
])

# 5. Criação da Corrente usando a Sintaxe LCEL (LangChain Expression Language) [cite: 10, 29, 30]
chain = prompt_template | llm

# 6. Gerenciamento de Memória Persistente do LangChain + Streamlit [cite: 29, 32]
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Caso o histórico esteja vazio, adiciona a mensagem inicial de acolhimento do assistente [cite: 32]
if len(msgs.messages) == 0:
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")

# Envelopando a nossa chain para gerenciar automaticamente a persistência do histórico [cite: 32]
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs, 
    input_messages_key="input",
    history_messages_key="history"
)

# 7. Configuração da Barra Lateral (Requisito: Botão de limpar histórico) [cite: 33, 38]
st.sidebar.title("Configurações do App")
st.sidebar.text("Modelo: Zephyr-7B-Chat")
st.sidebar.text("Framework: LangChain LCEL")

if st.sidebar.button("🧹 Limpar Histórico de Conversa"): [cite: 38]
    msgs.clear() # Executa a limpeza da memória [cite: 38]
    msgs.add_ai_message("Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?")
    st.rerun()

# 8. Renderização da Interface do Chat [cite: 33, 36]
for msg in msgs.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role): [cite: 36]
        st.write(msg.content)

# 9. Fluxo de Entrada do Usuário e Resposta da IA em Tempo Real [cite: 33, 37]
if user_input := st.chat_input("Converse com o Companion..."): [cite: 37]
    # Exibe imediatamente a mensagem digitada pelo usuário [cite: 36]
    with st.chat_message("user"): [cite: 36]
        st.write(user_input)
    
    # Chama a execução da corrente persistente do LangChain [cite: 32]
    with st.chat_message("assistant"): [cite: 36]
        with st.spinner("Pensando com cuidado..."):
            config = {"configurable": {"session_id": "companion_session"}}
            response = chain_with_history.invoke({"input": user_input}, config=config) [cite: 32]
            
            # Pega o conteúdo de texto purificado da resposta do ChatHuggingFace
            answer_text = response.content
            
            # Renderiza a resposta final do modelo na tela
            st.write(answer_text)
