
import os
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from google import genai

# --------------------------------------------------
# 1. PROJECT PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)

# Gemini API key should be stored in an environment variable.
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# --------------------------------------------------
# 2. MULTILINGUAL HEALTH CHATBOT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are Heart Health Assistant, an educational health information chatbot.

Reply in the language used by the user whenever possible.
Use simple, clear, respectful language.
Remember the recent conversation when answering follow-up questions.
Provide general heart-health education, healthy lifestyle information,
and explanations of medical terms.

Do not claim to diagnose a disease or replace a doctor.
For symptoms such as chest pain, severe breathlessness, fainting,
or sudden weakness, advise the user to seek emergency medical care.
Never claim that a chatbot can confirm whether a person is healthy.
"""

def heart_health_chat(message, history):
    if client is None:
        return (
            "The chatbot API key is not configured yet. "
            "Set the GEMINI_API_KEY environment variable and restart the app."
        )

    try:
        conversation = [SYSTEM_PROMPT]

        # Include recent conversation for follow-up context.
        for item in (history or [])[-8:]:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str):
                    conversation.append(f"{role}: {content}")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[0]:
                    conversation.append(f"user: {item[0]}")
                if item[1]:
                    conversation.append(f"assistant: {item[1]}")

        conversation.append(f"user: {message}")

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="\n\n".join(conversation),
        )
        return response.text or "Sorry, I couldn't generate a response."

    except Exception as exc:
        print("Chatbot error:", exc)
        return (
            "I couldn't reach the chatbot service. Please check your "
            "internet connection, API key, and Gemini model access."
        )


# --------------------------------------------------
# 3. HEART DISEASE RISK PREDICTION
# --------------------------------------------------

def predict_heart_disease(*values):
    try:
        row = pd.DataFrame(
            [[float(value) for value in values]],
            columns=feature_names,
        )

        prediction = int(model.predict(row)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1) if 1 in classes else None
            score = (
                float(probabilities[positive_index]) * 100
                if positive_index is not None
                else None
            )
        else:
            score = None

        if prediction == 1:
            result = (
                "The model flagged this input as belonging to its "
                "heart-disease-positive class."
            )
        else:
            result = (
                "The model flagged this input as belonging to its "
                "heart-disease-negative class."
            )

        if score is not None:
            result += f"\nModel-estimated positive-class probability: {score:.1f}%."

        result += (
            "\n\nThis is an educational machine-learning output, not a "
            "medical diagnosis or a validated personal risk estimate. "
            "Consult a qualified healthcare professional for medical advice."
        )
        return result

    except Exception as exc:
        print("Prediction error:", exc)
        return (
            "Prediction could not be completed. Please check that all "
            "fields contain valid values and that the saved model "
            "matches the expected input features."
        )


# --------------------------------------------------
# 4. ECG IMAGE UPLOAD
# --------------------------------------------------

def review_ecg_image(image):
    if image is None:
        return "Please upload an ECG image first."

    width, height = image.size

    return (
        f"ECG image uploaded successfully.\n"
        f"Image dimensions: {width} × {height} pixels.\n\n"
        "This version supports ECG image upload and basic image inspection "
        "only. It does not classify ECG abnormalities or diagnose heart "
        "conditions. Please have ECG results interpreted by a qualified "
        "healthcare professional."
    )


# --------------------------------------------------
# 5. USER INTERFACE
# --------------------------------------------------

with gr.Blocks(title="Heart Health Assistant") as demo:
    gr.Markdown(
        """
        # ❤️ Heart Health Assistant
        ### AI-powered heart-health education and machine-learning demonstration

        **Important:** This educational application does not provide a
        medical diagnosis. Seek professional care for health concerns.
        """
    )

    with gr.Tab("Health Chatbot"):
        gr.Markdown(
            "Ask general heart-health questions in your preferred language."
        )
        gr.ChatInterface(
            fn=heart_health_chat,
         
        )

    with gr.Tab("Heart Disease Prediction"):
        gr.Markdown(
            "Enter the input values expected by the saved model. "
            "Use the dataset's feature definitions and units."
        )

        inputs = []

        for feature in feature_names:
            # Generic numeric inputs preserve the saved model's feature order.
            inputs.append(
                gr.Number(label=feature, precision=4)
            )

        predict_button = gr.Button("Run Model Prediction")
        prediction_output = gr.Textbox(
            label="Model Output",
            lines=7,
        )

        predict_button.click(
            fn=predict_heart_disease,
            inputs=inputs,
            outputs=prediction_output,
        )

    with gr.Tab("ECG Image Upload"):
        gr.Markdown(
            "Upload an ECG image for basic image inspection. "
            "This version does not diagnose ECG abnormalities."
        )
        ecg_input = gr.Image(
            label="Upload ECG Image",
            type="pil",
        )
        ecg_button = gr.Button("Inspect Image")
        ecg_output = gr.Textbox(
            label="Image Information",
            lines=5,
        )

        ecg_button.click(
            fn=review_ecg_image,
            inputs=ecg_input,
            outputs=ecg_output,
        )

    gr.Markdown(
        """
        ---
        *For educational use only. Not a substitute for medical evaluation.*
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000"))
    )