# frontend.py
import gradio as gr
from chatbot import initialize_chatbot
from chatbot import qa
import requests

def store_feedback(question, answer, feedback):
    data = {
        "question": question,
        "answer": answer,
        "feedback": feedback
    }
    response = requests.post("http://localhost:8000/feedback", json=data)
    return response.json()

print(f'QA in frontend: {qa}')

def qa_chain(user_message, history):
    global qa
    history = history or []
    response = qa({"question": user_message, "chat_history": history})
    history.append((user_message, response["answer"]))
    return history,history, ""

# Front end web app
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

with gr.Blocks() as demo:
    gr.Markdown("## NeMo Chatbot")
    chatbot = gr.Chatbot()
    state = gr.State()
    #clear = gr.Button("Clear")
    # frontend.py continued
    txt = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.").style(container=False)
    with gr.Row():
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
    
    btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn]
    txt.submit(qa_chain, inputs=[txt, state], outputs=[chatbot, state, txt])
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    upvote_btn.click(store_feedback, inputs=[txt, chatbot, state], outputs=[upvote_btn, downvote_btn])
    downvote_btn.click(store_feedback, inputs=[txt, chatbot, state], outputs=[upvote_btn, downvote_btn])
