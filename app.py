import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace
from transformers import AutoTokenizer, pipeline
from keras.models import load_model
import joblib
import google.generativeai as genai
import pandas as pd

GEMINI_KEY = "AIzaSyDUEJe0asR2IMx2KytiRw-BehJuwo0Utq8"
genai.configure(api_key=GEMINI_KEY)

st.set_page_config(
    page_title="EngajaAula",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%);
}

/* Título grande */
.big-title {
    font-size: 52px !important;
    text-align: center;
    color: #0d47a1;
    font-weight: 900;
    margin-bottom: -10px;
    letter-spacing: -1px;
}

/* Subtitulo */
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #444;
    margin-bottom: 40px;
}

/* Cards com glass effect */
.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    margin-bottom: 25px;
    animation: fadein 0.6s ease;
}

/* Card amarelo da recomendação */
.rec-card {
    background: rgba(255, 243, 205, 0.65);
    border-left: 8px solid #f9a825;
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

/* Botão estilizado */
.stButton>button {
    background: linear-gradient(135deg, #1976d2, #0d47a1);
    border: none;
    color: white;
    font-size: 20px;
    font-weight: 600;
    padding: 18px;
    border-radius: 14px;
    width: 100%;
    transition: 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #1565c0, #0b3c91);
}

/* Fade animado */
@keyframes fadein {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0px); }
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>EngajaAula</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Análise multimodal de engajamento do aluno em tempo real</p>", unsafe_allow_html=True)

@st.cache_resource
def carregar_tudo():
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    emotion_pipe = pipeline("text-classification", model="pysentimiento/bert-pt-emotion")
    sentiment_pipe = pipeline("sentiment-analysis", model="lipaoMai/bert-sentiment-model-portuguese")

    model = load_model("modelo_engajamento_final.h5")
    scaler = joblib.load("scaler.pkl")

    gemini = genai.GenerativeModel("gemini-2.5-flash")
    return tokenizer, emotion_pipe, sentiment_pipe, model, scaler, gemini

tokenizer, emotion_pipe, sentiment_pipe, clf, scaler, gemini = carregar_tudo()


def preprocess_text(text):
    stop = ["a","o","e","é","de","do","da","em","um","para","com","não","uma","os","no","se","na",
            "por","mais","as","dos","como","mas","foi","ao","ele","das","tem","à","seu","sua","ou",
            "ser","quando","muito","nos","já","está","eu","também","só","pelo","pela","até","isso",
            "ela","entre","era","depois","sem","mesmo","você","que","nas","me","esse","eles",
            "estão","vai","vão","te"]

    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = [w for w in text.split() if w not in stop]
    return " ".join(words) if words else text


def get_text_features(text):
    clean = preprocess_text(text)
    sent = sentiment_pipe(clean)[0]
    emo = emotion_pipe(clean)[0]

    sent_vec = np.zeros(3)
    mapping = {"POSITIVE":0, "POS":0, "NEUTRAL":1, "NEU":1, "NEGATIVE":2, "NEG":2}
    sent_vec[mapping.get(sent["label"].upper(), 1)] = sent["score"]

    emo_onehot = np.zeros(7)
    emo_map = {'joy':0, 'sadness':1, 'anger':2, 'fear':3, 'surprise':4, 'disgust':5, 'others':6}
    emo_onehot[emo_map.get(emo["label"].lower(), 6)] = emo["score"]

    dummy_bert = np.zeros(1536)
    return np.concatenate([dummy_bert, sent_vec, emo_onehot])


def extract_facial(img_path):
    emb = DeepFace.represent(img_path, model_name="ArcFace",
                             enforce_detection=False,
                             detector_backend="skip")[0]["embedding"]

    emb = np.array(emb)
    emb = emb[:512] if len(emb) > 512 else np.pad(emb, (0, 512 - len(emb)))

    emo = DeepFace.analyze(img_path, actions=['emotion'],
                           enforce_detection=False,
                           detector_backend="skip",
                           silent=True)[0]["emotion"]

    emo_vec = np.array([
        emo['angry'], emo['disgust'], emo['fear'], emo['happy'],
        emo['sad'], emo['surprise'], emo['neutral']
    ]) / 100.0

    return np.concatenate([emb, emo_vec])



col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📸 Foto do aluno")
    foto = st.file_uploader("Enviar foto", type=["jpg", "jpeg", "png", "webp"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💬 O que o aluno disse?")
    texto = st.text_area("Comentário do aluno", height=140, placeholder="Ex: Tô achando difícil hoje...")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔍 Analisar Engajamento Agora"):
    if not foto:
        st.error("Por favor, envie a foto do aluno.")
    elif not texto.strip():
        st.error("Por favor, digite uma frase.")
    else:
        with st.spinner("🔄 Analisando rosto e texto..."):

            img = Image.open(foto)
            img.save("temp.jpg")

            st.image(img, width=320, caption="Aluno em análise")

            facial = extract_facial("temp.jpg")
            textual = get_text_features(texto)

            X = np.hstack([facial, textual]).reshape(1, -1)
            X_scaled = scaler.transform(X)

            probs = clf.predict(X_scaled)[0]
            pred = np.argmax(probs)

            nivel = ["Baixo", "Médio", "Alto"][pred]
            confiança = float(probs.max())

            emo_vec = facial[512:]
            emo_idx = np.argmax(emo_vec)
            emocoes = ["raiva", "nojo", "medo", "feliz", "triste", "surpresa", "neutro"]
            emo_dom = emocoes[emo_idx]

            st.markdown(f"""
            <div class='card'>
                <h2 style='text-align:center;'>🎯 Engajamento: <b>{nivel.upper()}</b></h2>
                <h4 style='text-align:center;'>
                    {confiança:.1%} de confiança • Emoção dominante: 
                    <b>{emo_dom.upper()}</b> ({emo_vec[emo_idx]:.1%})
                </h4>
            </div>
            """, unsafe_allow_html=True)

            df_bar = pd.DataFrame({"Nível": ["Baixo", "Médio", "Alto"], "Probabilidade": probs})
            st.bar_chart(df_bar.set_index("Nível"))

            with st.spinner("🧠 Gerando recomendação pedagógica..."):
                prompt = f"""
                Você é um pedagogo experiente.
                O aluno disse: "{texto}"
                Emoção facial dominante: {emo_dom} ({emo_vec[emo_idx]:.1%})
                Nível de engajamento previsto: {nivel} ({confiança:.1%})

                Gere uma recomendação curta (3 frases), prática e direta
                para o professor agir agora e melhorar o engajamento do aluno.
                """

                response = gemini.generate_content(prompt)
                rec = response.text.strip()

                st.markdown("<div class='rec-card'>", unsafe_allow_html=True)
                st.markdown("### 📝 Recomendação ao Professor")
                st.write(rec)
                st.markdown("</div>", unsafe_allow_html=True)
