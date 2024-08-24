import logging
import asyncio
from pathlib import Path
from pydub import AudioSegment
import gradio as gr
import torch
from TTS.api import TTS
from tqdm import tqdm
import fitz  # PyMuPDF
from newspaper import Article

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_text_from_url(url):
    """Fetch main text from the provided URL using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Failed to fetch text from URL: {e}")
        return None

def convert_pdf_to_text(pdf_path):
    """Convert PDF file to text using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

from ollama import Client

def run_ollama(prompt, model="phi3.5"):
    """Run Ollama locally with the given model and prompt using the Python API."""
    client = Client()
    logging.info(f"Running Ollama with model: {model} and prompt: {prompt}")
    try:
        response = client.generate(model=model, prompt=prompt)
        output = response['response']
        logging.info(f"Ollama response: {output}")
        return output
    except Exception as e:
        logging.error(f"Ollama error: {str(e)}")
        return None

def generate_prompt(language, stage):
    """Generate the appropriate prompt based on the language and stage."""
    if language.lower() == "english":
        return (
            "English Version:\n\n"
            "Generate an in-depth and coherent interview in dialogue format that reflects the key aspects of the provided document. "
            "Include a brief introduction by the interviewer, followed by a series of questions and responses, concluding with a summary."
            " Output should be plain text, with each dialogue line separated by two new lines."
        )
    else:
        return (
            "Versión en Español:\n\n"
            "Genera una entrevista coherente en formato de diálogo que refleje los aspectos clave del documento proporcionado. "
            "Incluye una breve introducción por el entrevistador, seguida de una serie de preguntas y respuestas, concluyendo con un resumen."
            " El resultado debe ser texto plano, con cada línea de diálogo separada por dos nuevas líneas."
        )

def get_chat_response(text, language):
    """Generate interview based on text and handle response."""
    prompt_stage = generate_prompt(language, 1)
    interview = run_ollama(prompt_stage + "\n\n" + text)
    return interview.split('\n\n')  # Splitting by two new lines as per the new format

# Setup TTS using 🐸TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def remove_prefix(text):
    """Remove any prefix before and including the first colon, if present."""
    index = text.find(':')
    if index != -1:
        return text[index + 1:].lstrip()
    return text

def split_long_sentence(sentence, max_length=230, max_pauses=8):
    """Splits a sentence into smaller parts if it's too long or has too many pauses."""
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        if possible_splits:
            split_at = possible_splits[-1] + 1
        else:
            split_at = max_length
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    parts.append(sentence)
    return parts

def generate_audio(dialogue, is_interviewer, output_dir):
    """Generate audio from a dialogue line using 🐸TTS after chunking it."""
    cleaned_text = remove_prefix(dialogue)
    chunks = split_long_sentence(cleaned_text)
    for index, chunk in enumerate(chunks):
        speaker_wav = "Interviewer.mp3" if is_interviewer else "Female.wav"
        speaker_name = "Interviewer" if is_interviewer else "Female"
        logging.info(f"Generating audio for chunk {index + 1} of {speaker_name}")
        speech_file_path = output_dir / f"speech_{speaker_name}_{index + 1}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=speaker_wav, language="en", file_path=str(speech_file_path))
    return [output_dir / f"speech_{speaker_name}_{i + 1}.wav" for i in range(len(chunks))]

def combine_wav_files(chapter_files, output_path):
    """Combine WAV files into a single file."""
    combined_audio = AudioSegment.empty()
    for chapter_file in chapter_files:
        audio_segment = AudioSegment.from_wav(chapter_file)
        combined_audio += audio_segment
    combined_audio.export(output_path, format='wav')
    print(f"Combined audio saved to {output_path}")

async def generate_and_combine_audio_files(dialogues, output_dir, base_name):
    """Generate audio files for dialogues and combine them."""
    chapter_files = []
    is_interviewer = True  # Start with interviewer as the first speaker
    for dialogue in tqdm(dialogues, desc="Generating audio"):
        if dialogue.strip():  # Check if there is actual dialogue content
            speech_files = generate_audio(dialogue, is_interviewer, output_dir)
            chapter_files.extend(speech_files)
            is_interviewer = not is_interviewer  # Toggle speaker after each dialogue block
    combined_audio_path = output_dir / f"{base_name}.wav"
    combine_wav_files(chapter_files, combined_audio_path)
    return combined_audio_path

async def main_async(input_data, language):
    """Main function to process input and generate audio."""
    text = ""
    if isinstance(input_data, Path):
        text = convert_pdf_to_text(input_data)
    else:
        text = fetch_text_from_url(input_data)
    dialogues = get_chat_response(text, language)
    base_output_dir = Path("Working_files")
    audio_output_dir = base_output_dir / "audio_files"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    combined_audio_path = await generate_and_combine_audio_files(dialogues, audio_output_dir, "combined_interview")
    final_output_dir = base_output_dir / "final_output"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    combined_audio_path.rename(final_output_dir / combined_audio_path.name)
    return combined_audio_path

def gradio_interface(input_file, url, language):
    """Gradio interface to process input and generate audio."""
    input_data = input_file if input_file else url
    try:
        audio_file_path = asyncio.run(main_async(input_data, language))
        return audio_file_path
    except Exception as e:
        logging.error(f"{e}")
        return str(e)

# Setup Gradio interface
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF / Subir PDF", type="filepath"),
        gr.Textbox(label="Or Enter Article URL", placeholder="Enter URL here"),
        gr.Dropdown(label="Select Language / Seleccionar idioma", choices=["English", "Spanish"], value="English")
    ],
    outputs=gr.Audio(label="Generated Interview / Entrevista generada"),
    allow_flagging="never"
)

# Launch Gradio interface
demo.launch(share=False)  # Set share=True to create a public link
