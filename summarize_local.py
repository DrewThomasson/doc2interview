print("starting...")

import os
import shutil
import subprocess
import re
from pydub import AudioSegment
import tempfile
from pydub import AudioSegment
import os
import nltk
from nltk.tokenize import sent_tokenize
import sys
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
import gradio as gr
from gradio import Progress
import urllib.request
import zipfile



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
from ollama import Client

import os





sample_ollama_response = """
Interviewer: Good day, Andrew Cunningham! We understand that you've delved into Microsoft’s Control Panel history and its potential future direction in line with the introduction of new operating systems such as Windows 10 and beyond. Can we explore this topic further? Thank you for joining us today.

Andrew: Absolutely, I believe it is quite an intriguing journey to trace back through Microsoft's user interface history! The Control Panel has been a fundamental part of the operating system experience since Windows NT 4 in 1996 – that’s more than three decades old.

Interviewer: That's right, Andrew. And with these years comes plenty of evolution and change for such an essential tool. The Control Panel provided a centralized location to view system settings which have been adjusted over the course of Windows history through various applets that users could interact with – from setting up hardware configurations down to network connections...

Andrew: Yes, indeed! Many elements in our day-to-day life are tied to these panels. For example, anyone who's used a computer for even an hour has probably adjusted the system date and time or changed their display settings at least once using Control Panel applets…

Interviewer: With that said, Microsoft’s recent note on support sites indicates it might be phasing out these traditional panels in favor of something more modern. What are your thoughts about this move? The company seems to promote a shift towards the Settings App which offers streamlined functionality and is tailored for touchscreen devices...

Andrew: It's an interesting development, I must admit! Microsoft has been moving forward with its Windows operating systems by introducing more modern interfaces. Although we still see older Control Panel applets in updates like 24H2, the fact remains that these traditional panels are being deprecated for newer alternatives – a sign of progress indeed…

Interviewer: But some users may have an attachment to these old designs and icons which date back as far as Windows Vista with its rounded glassy appearance. How significant is this change in design language, considering Microsoft’s attempt at keeping the interface cohesive? The company's modernization efforts are certainly visible throughout newer apps like Paint or Notepad…

Andrew: Yes, that was indeed a remarkable shift! It does pose questions about preserving legacy aspects of Windows. However, as operating systems mature and evolve with time to align better with the current technology landscape – touchscreens being more common now than ever before - these changes seem almost logical...

Interviewer: In summary Andrew, we can see that Microsoft has a significant history attached to its Control Panel interface. The company is progressively transitioning towards newer and streamlined interfaces while not completely neglecting the older styles for some user preferences… Any final thoughts on what you believe this might mean going forward?

Andrew: Well it’s always challenging when such a profound component of our daily interactions with technology changes. I suspect Microsoft will aim to balance new advancements and legacy features, as they've done previously – offering more contemporary interfaces without completely disregarding the older ones that many users might still prefer...

Interviewer: Thank you for your insights today, Andrew Cunningham! It’s been illuminating discussing Microsoft Windows evolution with a seasoned observer like yourself. Good day and hope to chat again soon!

Andrew: The pleasure was all mine – I always enjoy our talks on such fascinating topics too; catch you later!
"""





logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def wipe_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all the items in the given folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # If it's a file, remove it and print a message
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Removed file: {item_path}")
        # If it's a directory, remove it recursively and print a message
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory and its contents: {item_path}")
    
    print(f"All contents wiped from {folder_path}.")

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
    #interview = sample_ollama_response
    return interview.split('\n\n')  # Splitting by two new lines as per the new format

# Setup TTS using 🐸TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
#tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def remove_prefix(text):
    """Remove any prefix before and including the first colon, if present."""
    index = text.find(':')
    if index != -1:
        return text[index + 1:].lstrip()
    return text


def remove_prefix_from_all_txt_files_in_folder(folder_path):
    """Remove any prefix before and including the first colon in every .txt file in the specified folder."""
    
    def remove_prefix(text):
        """Remove any prefix before and including the first colon, if present."""
        index = text.find(':')
        if index != -1:
            return text[index + 1:].lstrip()
        return text

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                content = file.readlines()
            
            # Apply remove_prefix to each line
            new_content = [remove_prefix(line) for line in content]
            
            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_content)

    print("Prefix removed from all text files in the folder.")

# Usage example
#folder_path = '/path/to/your/folder'  # Replace with your folder path
#remove_prefix_from_folder(folder_path)


def create_chapter_files(chapters, output_folder):
    # Ensure the output directory exists, create if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    
    for i, chapter in enumerate(chapters, start=1):
        file_path = os.path.join(output_folder, f"chapter_{i}.txt")
        with open(file_path, "w") as file:
            file.write(chapter)

# Combine WAV files into a single file
def combine_wav_files(input_directory, output_directory, file_name):
    # Ensure that the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, file_name)

    # Initialize an empty audio segment
    combined_audio = AudioSegment.empty()

    # Get a list of all .wav files in the specified input directory and sort them
    input_file_paths = sorted(
        [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".wav")],
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    # Sequentially append each file to the combined_audio
    for input_file_path in input_file_paths:
        audio_segment = AudioSegment.from_wav(input_file_path)
        combined_audio += audio_segment

    # Export the combined audio to the output file path
    combined_audio.export(output_file_path, format='wav')

    print(f"Combined audio saved to {output_file_path}")

# Function to split long strings into parts
def split_long_sentence(sentence, max_length=230, max_pauses=8):
    """
    Splits a sentence into parts based on length or number of pauses without recursion.
    
    :param sentence: The sentence to split.
    :param max_length: Maximum allowed length of a sentence.
    :param max_pauses: Maximum allowed number of pauses in a sentence.
    :return: A list of sentence parts that meet the criteria.
    """
    parts = []
    while len(sentence) > max_length or sentence.count(',') + sentence.count(';') + sentence.count('.') > max_pauses:
        possible_splits = [i for i, char in enumerate(sentence) if char in ',;.' and i < max_length]
        if possible_splits:
            # Find the best place to split the sentence, preferring the last possible split to keep parts longer
            split_at = possible_splits[-1] + 1
        else:
            # If no punctuation to split on within max_length, split at max_length
            split_at = max_length
        
        # Split the sentence and add the first part to the list
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    
    # Add the remaining part of the sentence
    parts.append(sentence)
    return parts


#This function goes through the chapter dir and genrates a chapter for each chapter_1.txt and so on files
def convert_chapters_to_audio_standard_model(chapters_dir, output_audio_dir, target_voice_path=None, language=None):
    selected_tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(selected_tts_model, progress_bar=False).to(device)

    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)
    Narrerator_status = True

    for chapter_file in sorted(os.listdir(chapters_dir), key=lambda x: int(re.search(r"chapter_(\d+).txt", x).group(1)) if re.search(r"chapter_(\d+).txt", x) else float('inf')):
        if chapter_file.endswith('.txt'):
            match = re.search(r"chapter_(\d+).txt", chapter_file)
            if match:
                chapter_num = int(match.group(1))
            else:
                print(f"Skipping file {chapter_file} as it does not match the expected format.")
                continue

            chapter_path = os.path.join(chapters_dir, chapter_file)
            output_file_name = f"audio_chapter_{chapter_num}.wav"
            output_file_path = os.path.join(output_audio_dir, output_file_name)
            temp_audio_directory = os.path.join(".", "Working_files", "temp")
            os.makedirs(temp_audio_directory, exist_ok=True)
            temp_count = 0

            with open(chapter_path, 'r', encoding='utf-8') as file:
                chapter_text = file.read()
                sentences = sent_tokenize(chapter_text, language='italian' if language == 'it' else 'english')
                for sentence in tqdm(sentences, desc=f"Chapter {chapter_num}"):
                    fragments = split_long_sentence(sentence, max_length=249 if language == "en" else 213, max_pauses=10)
                    for fragment in fragments:
                        if fragment != "":
                            print(f"Generating fragment: {fragment}...")
                            fragment_file_path = os.path.join(temp_audio_directory, f"{temp_count}.wav")
                            #speaker_wav_path = target_voice_path if target_voice_path else default_target_voice_path
                            language_code = language if language else default_language_code
                            if Narrerator_status == True:
                                tts.tts_to_file(text=fragment, file_path=fragment_file_path, speaker_wav="Interviewer.mp3", language=language_code)
                            if Narrerator_status == False:
                                tts.tts_to_file(text=fragment, file_path=fragment_file_path, speaker_wav="Female.wav", language=language_code)
                            temp_count += 1

            combine_wav_files(temp_audio_directory, output_audio_dir, output_file_name)
            wipe_folder(temp_audio_directory)
            print(f"Converted chapter {chapter_num} to audio.")
            #This will swap the status of the Narrerator status boolean value
            Narrerator_status = not Narrerator_status

async def generate_and_combine_audio_files(dialogues, output_dir, base_name):
    """Generate audio files for dialogues and combine them."""
    file_number = 1  # Start numbering from 0000001
    is_interviewer = True  # Start with interviewer as the first speaker
    for dialogue in tqdm(dialogues, desc="Generating audio"):
        if dialogue.strip():  # Check if there is actual dialogue content
            generate_audio()
            print(f"Generating audio...: Interviewer is : {is_interviewer} dialogue is {dialogue}")
            is_interviewer = not is_interviewer  # Toggle speaker after each dialogue block
    combined_audio_path = output_dir / f"{base_name}.wav"
    print(f"combining audio files...")
    combine_audio()
    return combined_audio_path

async def main_async(input_data, language):
    """Main function to process input and generate audio."""
    text = ""
    if isinstance(input_data, Path):
        text = convert_pdf_to_text(input_data)
    else:
        text = fetch_text_from_url(input_data)
    dialogues = get_chat_response(text, language)
    #create chapter files from dialog
    chaptertxt_folder = "chapters_txt"
    create_chapter_files(dialogues, chaptertxt_folder)

    #This will remove all the prefix from all the txt files in the chaptertxt_folder folder
    remove_prefix_from_all_txt_files_in_folder(chaptertxt_folder)

    #generate audio for all chapter files
    output_audio_dir = "output_audio"
    convert_chapters_to_audio_standard_model(chaptertxt_folder, output_audio_dir, target_voice_path=None, language='en')

    #combine all the audio files into a single final output audio file
    final_output_audio_dir = "final_output_audio_dir"
    combine_wav_files(output_audio_dir, final_output_audio_dir, "final_output_audio.wav")

    #wipe all the temp folders
    wipe_folder("Working_files")
    wipe_folder("Working_files/temp")
    wipe_folder("output_audio")
    wipe_folder("chapters_txt")


    return "Complete!"

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
