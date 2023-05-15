# frontend.py
import gradio as gr
import time
import argparse
import json
import datetime
import os
from gradio_css import code_highlight_css
from lazy_model import LazyModel, LazyEmbedding
from lazy_model import LazyModel, LazyEmbedding
import yaml
from chatbot import initialize_chatbot

qa = None
current_model = None
current_embedding = None

# Load the configurations for the models and embeddings
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

models_config = config["models"]
embeddings_config = config["embeddings"]

# Create LazyModel and LazyEmbedding instances for each model and embedding
# models = {name: LazyModel(cfg) for name, cfg in models_config.items()}
# embeddings = {name: LazyEmbedding(cfg) for name, cfg in embeddings_config.items()}
models = {}
for model_config in models_config:
    model_name = model_config['name']
    models[model_name] = LazyModel(model_config)
embeddings = {}
for embedding_config in embeddings_config:
    embedding_name = embedding_config['name']
    embeddings[embedding_name] = LazyEmbedding(embedding_config)

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

block_css = code_highlight_css + """
pre {
  white-space: pre-wrap;       /* Since CSS 2.1 */
  white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
  white-space: -pre-wrap;      /* Opera 4-6 */
  white-space: -o-pre-wrap;    /* Opera 7 */
  word-wrap: break-word;       /* Internet Explorer 5.5+ */
  overflow-x: auto;            /* Add horizontal scrollbar if needed */
  overflow-y: hidden;          /* Prevent vertical scrollbar */
}
"""


'''following code is based on https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/gradio_web_server.py'''


# get user's interaction chat logs and store them in a json file
def get_conv_log_filename():
    t = datetime.datetime.now()
    dir_name = "chat_logs"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    name = os.path.join(dir_name, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def vote_last_response(state, vote_type, model_selector, embedding_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "embedding": embedding_selector,
            "state": state,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

def upvote_last_response(state, model_selector, embedding_selector, request: gr.Request):
    vote_last_response(state, "upvote", model_selector, embedding_selector, request)
    return ("",) + (disable_btn,) * 3

def downvote_last_response(state, model_selector, embedding_selector, request: gr.Request):
    vote_last_response(state, "downvote", model_selector, embedding_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, embedding_selector, request: gr.Request):
    vote_last_response(state, "flag", model_selector, embedding_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, model_selector, embedding_selector, request: gr.Request):
    last_user_message = state[-1][0]
    history = state[:-1]
    # Generate a new response by calling qa_chain.
    new_history, _, _, btn1, btn2, btn3, btn4, btn5 = qa_chain(last_user_message, history, model_selector, embedding_selector)
    vote_last_response(new_history, "regenerate", model_selector, embedding_selector, request)
    state = new_history
    return (state, state, "") + (enable_btn,) * 5

def clear_history():
    state = None
    return (state, [], "") + (disable_btn,) * 5

def qa_chain(user_message, history, model_selector, embedding_selector):
    global qa, current_model, current_embedding
    history = history or []

    # Initialize the selected chatbot if it's different from the current one
    if model_selector != current_model or embedding_selector != current_embedding:
        lazy_model = models[model_selector]
        lazy_embedding = embeddings[embedding_selector]
        qa = initialize_chatbot(lazy_model, lazy_embedding, config)
        current_model = model_selector
        current_embedding = embedding_selector

    response = qa({"question": user_message, "chat_history": history})
    metadata_list = [doc.metadata for doc in response["source_documents"]]
    # Remove duplicate sources
    unique_sources = list(set([meta['source'] for meta in metadata_list]))
    # Combine the answer and metadata list into a single string
    formatted_sources = '<br>'.join([f"source: {src}" for src in unique_sources])
    # Combine the answer and sources into a single string
    formatted_response = f"{response['answer']}<br><br>{formatted_sources}"
    history.append((user_message, formatted_response))
    return (history,history, "") + (enable_btn,) * 5


def build_single_model_ui(models, embeddings):
    '''create frontend UI for the chatbot'''
    # add disclaimer
    notice_markdown = """

### Choose a model to chat with
"""
    state = gr.State()
    gr.Markdown(notice_markdown, elem_id= "notice_markdown")

    # create select model dropdown block
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(models.keys()),
            value=list(models.keys())[0],
            interactive=True,
            show_label=False,
        ).style(container=False)
    
        embedding_selector = gr.Dropdown(
            choices=list(embeddings.keys()),
            value=list(embeddings.keys())[0],
            interactive=True,
            show_label=False,
        ).style(container=False)

    # create chatbot block and user input block 
    chatbot = gr.Chatbot(
        elem_id="chatbot", label="Scroll down and start chatting", visible=True
    ).style(height=500)
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=True,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=True)
    
    with gr.Row(visible=True) as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)


    learn_more_md = """to learn more """
    gr.Markdown(learn_more_md, elem_id="learn_more_md")

    '''define the logic of the UI buttons (register listeners)'''

    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    upvote_btn.click(
        upvote_last_response,
        [state, model_selector, embedding_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )

    downvote_btn.click(
        downvote_last_response,
        [state, model_selector, embedding_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )

    flag_btn.click(
        flag_last_response,
        [state, model_selector, embedding_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(
        regenerate,
        [state, model_selector, embedding_selector],
        [state, chatbot, textbox] + btn_list,
    )

    # when user clicks clear, it will clear the history and update the state and chatbot
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list, queue=False)

    # select new model will clear the history and update the state and chatbot
    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list, queue=False)

    # select new model will clear the history and update the state and chatbot
    embedding_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list, queue=False)
    
    # after user types in the text and press enter, it will add the text to the history and update the state and chatbot
    textbox.submit(qa_chain, inputs=[textbox, state, model_selector, embedding_selector], outputs=[state, chatbot, textbox] + btn_list)

    # after user clicks send button, it will add the text to the history and update the state and chatbot
    send_btn.click(qa_chain, inputs=[textbox, state, model_selector, embedding_selector], outputs=[state, chatbot, textbox] + btn_list)

    return state, model_selector, chatbot, textbox, send_btn, button_row


def build_demo(models, embeddings):
    with gr.Blocks(
        title="Chat with NeMo SME",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        build_single_model_ui(models, embeddings)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default="7860")
    parser.add_argument("--concurrency-count", type=int, default=3)
    args = parser.parse_args()

    demo = build_demo(models, embeddings)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_port=args.port, max_threads=200
    )
    #demo.launch()

if __name__ == "__main__":
    main()
