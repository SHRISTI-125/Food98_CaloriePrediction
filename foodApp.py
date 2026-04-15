import streamlit as st
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sentence_transformers import SentenceTransformer
import speech_recognition as sr

# title
st.set_page_config(page_title="🍃 Calorie Optimizer", layout="wide")

# css
st.markdown("""
<style>
.card {
    background: green;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.title {
    font-size: 24px;
    font-weight: bold;
    color: #16a34a;
}
.metric {
    font-size: 18px;
    font-weight: bold;
}
.tag {
    background-color: #dcfce7;
    color: #166534;
    padding: 6px 12px;
    border-radius: 10px;
    margin: 5px;
    display: inline-block;
}
.bad {
    background-color: #fee2e2;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:green;'>🍃 Calorie Optimizer</h1>", unsafe_allow_html=True)

# start
@st.cache_resource
def load_all():
    model = load_model("food_prediction_10epoches.h5")
    print("loaded model")

    with open("food_embeddings.pkl", "rb") as f:
        food_list, food_embeddings = pickle.load(f)

    nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("loaded nlp")

    with open("labels.json", "r") as f:
        labels = json.load(f)

    with open("foodData.json", "r") as f:
        food_data = json.load(f)

    food_dict = {item["foodname"]: item for item in food_data}

    return model, food_list, food_embeddings, nlp_model, labels, food_dict


model, food_list, food_embeddings, nlp_model, labels, food_dict = load_all()

# predicting

def predict_text(user_text):
    user_text = user_text.lower().replace("_", " ")
    user_emb = nlp_model.encode([user_text])[0]

    cos_sim = np.dot(food_embeddings, user_emb) / (
        np.linalg.norm(food_embeddings, axis=1) * np.linalg.norm(user_emb)
    )

    idx = np.argmax(cos_sim)
    return food_list[idx] if cos_sim[idx] >= 0.40 else None


def predict_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)
    pred = tf.nn.softmax(pred)

    idx = int(np.argmax(pred))
    return labels[str(idx)]


def fetch_details(food):
    return food_dict.get(food, None)


# ui
def show_card(food, details):
    st.markdown(f"""
    <div class="card">
        <div class="title">🍽 {food.replace("_"," ").title()}</div>
        <div class="metric"> {details.get("calories_per_piece", "N/A")} kcal</div>
        <p>{details.get("description","")}</p>
    </div>
    """, unsafe_allow_html=True)

    # Nutrients
    st.markdown("### Nutrients")
    cols = st.columns(3)
    nutrients = details.get("nutrients", {})

    for i, (k, v) in enumerate(nutrients.items()):
        cols[i].markdown(f"<div class='card'><b>{k.capitalize()}</b><br>{v}</div>", unsafe_allow_html=True)

    # Bad for
    if "bad_for" in details:
        st.markdown("### Not Recommended")
        for item in details["bad_for"]:
            st.markdown(f"<span class='tag bad'>{item}</span>", unsafe_allow_html=True)

    # Alternative
    if "alternative" in details:
        st.markdown("### Better Alternative")
        st.success(details["alternative"])



col1, col2 = st.columns([2, 1])

# sidebar for only food that can be predicted
st.sidebar.title("Supported Foods")
if st.sidebar.checkbox("Show Food List"):
    for food in sorted(food_dict.keys()):
        st.sidebar.write("•", food.replace("_", " ").title())

# ---------------- INPUT ----------------
with col1:
    option = st.radio("Choose Input:", ["Image", "Text", "Voice"], horizontal=True)

    if option == "Image":
        file = st.file_uploader("Upload Image", type=["jpg","png"])

        if file and st.button("Predict"):
            with st.spinner("Analyzing..."):
                food = predict_image(file)

    elif option == "Text":
        text = st.text_input("Enter Food")

        if st.button("Predict"):
            food = predict_text(text)

    elif option == "Voice":
        if st.button("Speak"):
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    st.info("Speak now...")
                    audio = r.listen(source)

                text = r.recognize_google(audio)
                st.success(f"You said: {text}")
                food = predict_text(text)

            except:
                st.error("Voice error")

# output
with col2:
    try:
        if food:
            details = fetch_details(food)

            if details:
                show_card(food, details)
            else:
                st.warning("No data found")
    except:
        pass