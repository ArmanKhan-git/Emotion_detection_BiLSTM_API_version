import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

st.title("🧠 Emotion Detection App")
st.write("Enter text and detect emotion using BiLSTM model")

# check api health
try:
    health = requests.get(f"{API_URL}/health").json()
    if health["status"] == "ok":
        st.success("API is running")
except:
    st.error("API not running. Start FastAPI server first.")

# user input
text = st.text_area(
    "Enter text",
    placeholder="Example: I am feeling very happy today",
    height=150
)

# predict button
if st.button("Predict Emotion"):

    if len(text.strip()) == 0:
        st.warning("Please enter text")
    else:
        payload = {"text": text}

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 200:
                result = response.json()

                st.subheader("Prediction")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Emotion", result["emotion"])

                with col2:
                    st.metric("Confidence", f'{result["confidence"]}%')

                st.progress(int(result["confidence"]))

            else:
                st.error(response.json())

        except Exception as e:
            st.error("Error connecting to API")
            st.write(e)