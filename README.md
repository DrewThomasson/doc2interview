# 🎙️ doc2interview

Welcome to the Interview Audio Generator! 
This  innovative tool automatically transforms PDF documents or online articles into engaging interview-style audio files. 
It's perfect for auditory learners or anyone who enjoys consuming content on the go!
And best of all it runs entirly locally on your computer! No paid api services or anything.

## 📋 Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Ollama with the `phi3.5` model pulled

- For faster results have a cuda capable machine so xtts can generate faster with a minimum of 4gb Vram


## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DrewThomasson/doc2interview.git
   cd doc2interview
   ```

2. **Install required Python packages**:
   - Install all necessary Python packages using the following command:
     ```bash
     pip install -r requirements.txt
     ```

3. **Setup Ollama**:
   - Install Ollama following the official documentation.
   - Pull the `phi3.5` model necessary for running the script:
     ```bash
     ollama pull phi3.5
     ```

## 🚀 Quick Start

1. **Start the script**:
   ```bash
   python summarize_local.py  # Replace with your actual script name
   ```
2. **Open the Gradio interface**:
   - The interface will be available in your web browser.
   - Upload a PDF or enter an article URL.
   - Choose the language and let the magic happen!

## 📁 Output

The generated audio files will be stored in:
- **Chapter-wise audio**: `./output_audio/`
- **Final combined audio**: `./final_output_audio_dir/final_output_audio.wav`

Feel free to explore the audio files and use them as needed!

## 🎧 Demo

Check out this sample audio from a generated interview:

https://github.com/user-attachments/assets/77e6046d-18e0-41dd-b034-7cdd709b9daf



## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## 📖 License

Distributed under the MIT License. See `LICENSE` for more information.

## ❓ Support

Got questions? Feel free to open an issue or contact me directly at your-email@example.com.

## 🌟 Show your support

Give a ⭐️ if this project helped you!

## Inspired by 
AiPeterWorld with his non-offline version which used gemini flash and openai voice for tts

https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue


