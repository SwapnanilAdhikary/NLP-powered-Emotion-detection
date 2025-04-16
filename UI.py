from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import gradio as gr

model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

def generate_reply(text, emotion):
    responses = {
        "joy": "I'm so happy to hear that! 😊",
        "anger": "I can sense some frustration. Want to talk more about it?",
        "sadness": "I'm here for you. It’s okay to feel sad sometimes. 💙",
        "fear": "That sounds scary. Do you want to talk about it?",
        "surprise": "Oh wow! That’s surprising! 😲",
        "disgust": "That doesn't sound pleasant. Want to share more?",
        "neutral": "Got it! Let’s keep chatting.",
    }
    return responses.get(emotion, "Thanks for sharing your thoughts!")

def advanced_chatbot(user_input, history):
    scores = classifier(user_input)[0]
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_scores[0]['label'].lower()
    confidence = sorted_scores[0]['score']

    bot_reply = generate_reply(user_input, top_emotion)
    full_reply = f"{bot_reply} _(Emotion: {top_emotion}, confidence: {confidence:.2f})_"

    return full_reply  

chatbot_ui = gr.ChatInterface(
    fn=advanced_chatbot,
    title="💬 Advanced Emotion Chatbot",
    description="This AI bot detects your emotion and responds empathetically.",
    theme="soft"
)

if __name__ == "__main__":
    chatbot_ui.launch()