# %%
import argparse
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

import gradio as gr
import pandasai as pai
from dotenv import load_dotenv
from loguru import logger
from pandasai_litellm.litellm import LiteLLM

load_dotenv()


def parse_args():
    parser = ArgumentParser(description="DataChat", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", default=False, help="Share the app")
    parser.add_argument("--auth", action="store_true", default=True, help="Enable authentication")
    parser.add_argument("--auth-message", type=str, default=None, help="Authentication message")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openrouter",
        help="LLM provider",
        choices=["openrouter", "vsellm", "polzaai"],
    )
    parser.add_argument("--llm-name", type=str, default="gpt-5-mini", help="LLM name")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL")
    parser.add_argument("--llm-enable-cache", action="store_true", default=False, help="Enable cache")
    parser.add_argument("--llm-conversational", action="store_true", default=True, help="Enable conversational")
    parser.add_argument("--llm-save-logs", action="store_true", default=True, help="Enable save logs")
    parser.add_argument("--llm-verbose", action="store_true", default=False, help="Enable verbose")
    parser.add_argument("--llm-max-retries", type=int, default=5, help="Max retries")

    args = parser.parse_args()
    return args


def fn_authenticate(username, password):
    if username == CONFIG.gradio_username and password == CONFIG.gradio_password:
        return True
    else:
        return False


class CONFIG:
    gradio_username = os.getenv("GRADIO_USERNAME", "sber")
    gradio_password = os.getenv("GRADIO_PASSWORD", "sber123")
    llm_provider = "openrouter"
    llm_name = "gpt-5-mini"
    llm_api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
    llm_base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL")
    llm: LiteLLM = None
    llm_enable_cache = False
    llm_conversational = True
    llm_save_logs = True
    llm_verbose = False
    llm_max_retries = 5
    datasets: List = None
    history = None
    message = None
    full_message = None
    response = None
    # app_css = ".gradio-container {background-color: grey}"
    app_css = None
    app_title = "DataChat"
    app_description = "DataChat is a chatbot that can help you with your data."
    app_share = False
    app_server_name = "0.0.0.0"
    app_server_port = 7860
    app_fn_auth = fn_authenticate
    app_auth_message = None
    app_queue_enabled = True
    app_queue_max_size = 100
    app_examples = [
        "Построй гистрограмму заболеваний сердца по возрастам",
        "Сколько погибло на титанике людей для тестовой и тренировочной выборок?",
        "Постройте столбцовую диаграмму зависимости выживаемости от возраста. Разбей на 10 групп. Используй не проценты, а количество людей",
        "Сколько мужчин и женщин в каждом Классе (Pclass)? Построй столбцовую диаграмму с полученной информацией",
        "Какое количество выживших мужчин и женщин после крушения? Постройте столбцовую диаграмму с полученной информацией",
        "Постройте столбцовую диаграмму зависимости смертности от возраста. Разбей на 10 групп",
    ]


def setup_llm(model_choice):
    previous_model_name = CONFIG.llm_name
    logger.info(f"Setting up LLM: {model_choice}")
    try:
        CONFIG.llm_name = model_choice
        CONFIG.llm = LiteLLM(
            model=CONFIG.llm_name,
            api_key=CONFIG.llm_api_key,
            base_url=CONFIG.llm_base_url,
        )
        pai.config.set(
            {
                "llm": CONFIG.llm,
                "enable_cache": CONFIG.llm_enable_cache,
                "conversational": CONFIG.llm_conversational,
                "save_logs": CONFIG.llm_save_logs,
                "verbose": CONFIG.llm_verbose,
                "max_retries": CONFIG.llm_max_retries,
            }
        )
    except Exception as e:
        CONFIG.llm_name = previous_model_name
        logger.error(f"Error setting up LLM: {e}")
    return CONFIG.llm_name


def load_datasets(dataset_paths):
    CONFIG.datasets = []
    if dataset_paths:  # проверка на None или пустой список
        for dataset_path in dataset_paths:
            df = pai.read_csv(dataset_path)
            CONFIG.datasets.append(df)


def get_full_message():
    full_message = ""
    if CONFIG.history is not None and isinstance(CONFIG.history, list) and len(CONFIG.history) > 0:
        for msg in CONFIG.history:
            role = msg.get("role")
            content = msg.get("content")
            full_message += f"role={role}: {[c.get(c.get('type', 'text'), 'text') for c in content]}\n"
    full_message += f"role=user: {CONFIG.message}\n"
    logger.info(f"full_message: {full_message}")
    return full_message


# %%
def predict(message, history):
    CONFIG.message = message
    CONFIG.history = history
    if CONFIG.datasets is not None and isinstance(CONFIG.datasets, list) and len(CONFIG.datasets) > 0:
        CONFIG.full_message = get_full_message()
        response = pai.chat(CONFIG.full_message, *CONFIG.datasets)
    else:
        return "No datasets loaded. Please upload datasets."
    CONFIG.response = response
    logger.info(f"response.type: {response.type}, response.value: {response.value}\n")
    if response.type.lower() == "chart":
        response.save("chart.png")
        return gr.Image(response.value)
    elif response.type.lower() == "string":
        return response.value
    elif response.type.lower() == "number":
        return str(response.value)
    elif response.type.lower() == "dataframe":
        return response.value.to_string()


# %%
with gr.Blocks(analytics_enabled=False, title="DataChat") as demo:
    gr.Markdown(f"# {CONFIG.app_title}")
    gr.Markdown(f"## {CONFIG.app_description}")
    with gr.Row(scale=1):
        if CONFIG.llm is None:
            setup_llm(CONFIG.llm_name)
        select_model_dropdown = gr.Dropdown(
            label="Select Model Name",
            choices=["gpt-5-mini", "gpt-5.2"],
            value=CONFIG.llm_name,
        )
        select_model_dropdown.change(
            fn=setup_llm,
            inputs=select_model_dropdown,
            outputs=None,
        )
    with gr.Row(scale=1):
        files_upload = gr.File(
            label="Upload Data Files",
            file_count="multiple",
            file_types=[".csv", ".xls", ".xlsx", ".xlsm", ".xlsb", ".feather"],
        )
        files_upload.change(
            fn=load_datasets,
            inputs=files_upload,
            outputs=None,
        )
        # Events.change,    # при любом изменении значения
        # Events.select,    # при выборе файла в списке
        # Events.clear,     # при очистке
        # Events.upload,    # после загрузки файла
        # Events.delete,    # при удалении файла
        # Events.download,  # при СКАЧИВАНИИ файла (не загрузке!)
    with gr.Row(scale=20, min_height=600):
        gr.ChatInterface(
            fn=predict,
            textbox=gr.Textbox(placeholder=CONFIG.app_examples[0]),
            editable=True,
            examples=CONFIG.app_examples,
            cache_examples=False,
        )


def main(args):
    CONFIG.app_server_name = args.server_name
    CONFIG.app_server_port = args.server_port
    CONFIG.app_share = args.share
    CONFIG.app_fn_auth = fn_authenticate if args.auth else None
    CONFIG.app_auth_message = args.auth_message
    CONFIG.llm_provider = args.llm_provider
    CONFIG.llm_name = args.llm_name
    CONFIG.llm_api_key = args.llm_api_key
    CONFIG.llm_base_url = args.llm_base_url
    CONFIG.llm_enable_cache = args.llm_enable_cache
    CONFIG.llm_conversational = args.llm_conversational
    CONFIG.llm_save_logs = args.llm_save_logs
    CONFIG.llm_verbose = args.llm_verbose
    CONFIG.llm_max_retries = args.llm_max_retries
    demo.launch(
        server_name=CONFIG.app_server_name,
        server_port=CONFIG.app_server_port,
        auth=CONFIG.app_fn_auth,
        auth_message=CONFIG.app_auth_message,
        css=CONFIG.app_css,
        share=CONFIG.app_share,
    )


def is_interactive():
    """True if running in interactive mode (ipython, jupyter, etc.)"""
    return hasattr(sys, "ps1")


if __name__ == "__main__":
    if not is_interactive():
        args = parse_args()
    else:
        args = argparse.Namespace(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            auth=True,
            auth_message=None,
            llm_provider="openrouter",
            llm_name="gpt-5-mini",
            llm_api_key=None,
            llm_base_url=None,
            llm_enable_cache=False,
            llm_conversational=True,
            llm_save_logs=True,
            llm_verbose=False,
            llm_max_retries=5,
        )

    main(args)
