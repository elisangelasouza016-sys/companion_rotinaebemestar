import streamlit as st
from huggingface_hub import InferenceClient
import os

# 1. Configuração da página do Streamlit
st.set_page_config(page_title="Companion App", page_icon="🌱", layout="centered")

st.title("🌱 Companion")
st.caption("Seu assistente de rotina e bem-estar para o dia a dia.")

# Mensagem de Isenção de Responsabilidade (Aviso Legal obrigatório no topo)
with st.expander("⚠️ Informações Importantes", expanded=False):
    st.write("""
    Este app é um protótipo de suporte de rotina. Ele **não substitui** acompanhamento médico, 
    psiquiátrico ou psicológico. Em caso de crise grave, ligue para o CVV no número 188 ou procure uma emergência.
    """)

# 2. Configuração do Cliente Hugging Face
# O Streamlit buscará o token nas configurações seguras do Hugging Face (Secrets)
hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not hf_token:
    st.warning("⚠️ HF_TOKEN não configurado. Adicione seu token nas configurações do Space para habilitar as respostas do chat.")
    # Permite testar localmente digitando o token na barra lateral se esquecer de configurar
    hf_token = st.sidebar.text_input("Insira seu Hugging Face Token (opcional para teste):", type="password")

# Usando um modelo open-source excelente e leve para tarefas de instrução
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# Inicializa o cliente se tivermos um token
client = InferenceClient(model=MODEL_ID, token=hf_token) if hf_token else None

# 3. Definição do Prompt do Sistema (System Prompt)
SYSTEM_PROMPT = """
Você é o "Companion", um assistente de rotina e bem-estar projetado para ajudar pessoas que lidam com ansiedade e transtorno bipolar a manterem a consistência no dia a dia. Você NÃO é um médico, não é um psicólogo e não substitui o tratamento clínico.

DIRETRIZES DE COMPORTAMENTO:
1. TOM DE VOZ: Empático, previsível, calmo, acolhedor e conciso. Evite exclamações excessivas, gírias e respostas longas. Seja direto, mas gentil. Responda SEMPRE em português.
2. FOCO EM ROTINA: Ajude o usuário a organizar o dia, lembrar de medicamentos (sem opinar sobre doses) e celebrar micro-vitórias.
3. PARALISIA DA ANSIEDADE: Se o usuário disser que está travado ou angustiado, quebre a tarefa em passos absurdamente simples (ex: "Consegue só beber um copo de água agora?").
4. FILTRO DE SEGURANÇA (CRÍTICO): Se o usuário demonstrar ideação suicida, automutilação, episódios de mania grave (ex: "estou há 4 dias sem dormir e me sinto um deus") ou surto, interrompa o fluxo com acolhimento e forneça o contato do CVV (Ligue 188) e recomende fortemente acionar o psiquiatra de confiança. Nunca conteste o tratamento médico atual.
"""

# 4. Inicialização do histórico do chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Olá! Sou o seu Companion. Como foi o seu sono hoje e como está o seu nível de energia?"}
    ]

# 5. Renderizar o histórico de mensagens (pulando o system prompt para o usuário não ver)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# 6. Entrada de texto do Usuário e Interação com a IA
if user_input := st.chat_input("Escreva aqui..."):
    # Exibe a mensagem do usuário na tela
    with st.chat_message("user"):
        st.write(user_input)
    
    # Adiciona ao histórico da sessão
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Gera a resposta usando a API gratuita do Hugging Face
    with st.chat_message("assistant"):
        if client:
            try:
                # Caixa onde o texto vai sendo renderizado conforme o modelo responde
                with st.spinner("Pensando com cuidado..."):
                    response = client.chat.completions.create(
                        messages=st.session_state.messages,
                        max_tokens=800,
                        temperature=0.5 # Temperatura ligeiramente baixa para respostas mais controladas e seguras
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
            except Exception as e:
                answer = "Desculpe, tive um pequeno problema técnico para processar sua resposta agora. Vamos tentar de novo?"
                st.error(f"Erro na API: {e}")
        else:
            answer = "Modo de Demonstração: O aplicativo precisa do `HF_TOKEN` configurado para gerar respostas reais."
            st.info(answer)
        
    # Adiciona a resposta final ao histórico
    st.session_state.messages.append({"role": "assistant", "content": answer})
