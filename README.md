# GPT-powered Text Summarizer

## Overview
This project features an AI-powered text summarizer designed to efficiently condense large documents into concise summaries. Utilizing advanced natural language processing techniques, it focuses on extracting key information while maintaining the core essence of the original text.

## Features
- **Supervised Learning Focused**: Tailored for summarizing texts on classification and regression in supervised learning.
- **Chunk-Based Summarization**: Breaks down large documents into manageable chunks to ensure detail preservation.
- **GPT-Powered Summaries**: Leverages OpenAI's GPT models for accurate and coherent summarization.
- **Customizable Summarization**: Supports various summarization styles, including keyword and concept-focused summaries.
- **Logging and Archives**: Maintains logs of GPT responses and archives outputs for reference.

## Requirements
- Python 3.x
- An [OpenAI](https://openai.com/) API key
- Required Python packages: 
  - `nltk`
  - `openai`
  - `pathlib`
  - `python-dotenv`
  - `requests`
  - `tqdm`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Setup
1. Clone the repository.
2. Ensure Python 3.x is installed on your system.
3. Install the required Python packages: `pip install -r requirements.txt`.
4. Set your OpenAI API key in an environment variable or modify `main.py` to include your API key.

## Usage
1. Place the text you want to summarize in `input.txt`.
2. Run the script: `python main.py`.
3. Find the summarized text in `output.txt`.
4. Check `gpt_logs` and `summarizer.log` for detailed logs and API responses.

## Customization
- Modify `CHUNK_SIZE` in `main.py` to adjust the size of text chunks for summarization.
- Change prompts in `main.py` to alter the summarization style as per your requirements.

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
