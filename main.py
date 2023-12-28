import os
import re
import nltk
import logging
import requests
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from time import time, sleep
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize


DRY_RUN = False  # Set to False to enable actual GPT API calls
DELIMITER = '\n"""\n'  # Delimiter used to separate text in GPT messages
MAX_CHUNKS = -1  # Set the max number of chunks to summarize (recommended: -1 for unlimited)
CHUNK_SIZE = 1000  # Size of each text chunk for summarization (recommended: 1,000 to 5,000)
GPT_REQUEST_TIMEOUT = 180  # Timeout for GPT API requests in seconds (recommended: >= 60s)
GPT_REQUEST_MAX_RETRY = 5  # Maximum number of retries for GPT API requests (recommended: 2 to 5)
FILE_IN = str(Path.cwd() / 'input.txt')  # Input file containing the original text
FILE_OUT = str(Path.cwd() / 'output.txt')  # Output file for the summarized text

# Initialize logging
logging.basicConfig(filename='summarizer.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Download the necessary NLTK models for splitting by sentence
nltk.download('punkt')

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'))


def open_file(filepath: str) -> str:
    """
    Opens and reads the content of a file.

    Args:
        filepath (str): Path to the file to be read.
    Returns:
        str: The content of the file.
    Raises:
        Exception: If the file cannot be opened or read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except Exception as _e:
        logging.error(f'Error opening file {filepath}: {_e}')
        raise


def save_file(content: str, filepath: str) -> None:
    """
    Saves content to a file.

    Args:
        content (str): Content to be saved.
        filepath (str): Path where the content will be saved.
    Raises:
        Exception: If the file cannot be written.
    """
    if not filepath:
        logging.error("File path is empty")
        raise ValueError("File path is empty")

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
    except Exception as _e:
        logging.error(f'Error saving file {filepath}: {_e}')
        raise


def remove_extra_whitespace(text: str) -> str:
    """
    Removes extra whitespaces from the text.

    Args:
        text (str): The original text.
    Returns:
        str: The text with extra whitespaces removed.
    """
    return re.sub(r'\s+', ' ', text.strip()).strip()


def chunk_text_by_sentences(text, max_chunk_size):
    """
    Splits the text into chunks, each containing whole sentences and having a length close to max_chunk_size.

    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): The approximate maximum size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    # Split the text into sentences
    sentences = sent_tokenize(text)

    _chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding the next sentence would exceed the max_chunk_size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # If so, add the current_chunk to the chunks list and start a new chunk
            _chunks.append(current_chunk)
            current_chunk = sentence
        else:
            # Otherwise, add the sentence to the current chunk
            current_chunk += " " + sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        _chunks.append(current_chunk)

    return _chunks


def summarize_with_gpt(_text: str, _model: str = 'gpt-3') -> str:
    """
    Summarizes text using the GPT model.

    Implements an exponential backoff strategy for retries in case of request failures.

    Args:
        _text (str): Text to be summarized.
        _model (str, optional): The GPT model to be used. Defaults to 'gpt-3'.
    Returns:
        str: Summarized text or error message in case of failure.
    """
    if DRY_RUN:
        # Return the original text to simulate a summary in dry-run mode
        return remove_extra_whitespace(_text)

    if not _model or _model == 'gpt-3':
        _model = 'gpt-3.5-turbo'
    if _model == 'gpt-4':
        _model = 'gpt-4-1106-preview'

    _messages = [
        {
            "role": "system",
            "content": "You are a writing assistant, skilled in revising and summarizing complex technical writing "
                       "with accuracy and precision."
        },
        {
            "role": "user",
            "content": "Provide an executive summary of the following text (delimited by triple quotes). "
                       "Present the key ideas and findings directly, without bullet points, "
                       "as if for a busy professional who needs to grasp the essential points quickly. "
                       "Ignore complete sentences and grammatical correctness. "
                       "Abbreviate long and repetitive words. "
                       f'{DELIMITER}{_text}{DELIMITER}'
        }
    ]

    err = None
    retry = 0
    retry_delay = 1  # Initial delay in seconds

    while retry < GPT_REQUEST_MAX_RETRY:
        try:
            # Send a request to the GPT API
            response = client.chat.completions.create(model=_model, messages=_messages, timeout=GPT_REQUEST_TIMEOUT)
            text = remove_extra_whitespace(response.choices[0].message.content)
            log_text = 'PROMPT:\n\n' + _text + '\n\n==========\n\nRESPONSE:\n\n' + text
            save_file(log_text, f'gpt_logs/{time()}_gpt.txt')
            return text
        except requests.exceptions.RequestException as req_err:
            logging.error(f'Network-related error on try {retry + 1}/{GPT_REQUEST_MAX_RETRY}: {req_err}')
        except Exception as err:
            logging.error(f'General GPT request error on try {retry + 1}/{GPT_REQUEST_MAX_RETRY}: {err}')
        finally:
            retry += 1

        if retry >= GPT_REQUEST_MAX_RETRY:
            logging.error(f'GPT request failed after {GPT_REQUEST_MAX_RETRY} retries')
            return f'GPT error: {err}' if err else 'GPT error: Unknown error'
        else:
            sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff


if __name__ == '__main__':
    try:
        # Read the original text from the input file
        original_text = open_file(FILE_IN)

        # Split the original text into chunks
        chunks = chunk_text_by_sentences(original_text, CHUNK_SIZE)

        if MAX_CHUNKS and MAX_CHUNKS > 0:
            chunks = chunks[:MAX_CHUNKS]

        count = 0
        result = []

        for chunk in tqdm(chunks, desc='Summarizing chunks', leave=True):
            count += 1
            summary = summarize_with_gpt(chunk, _model='gpt-4')
            result.append(summary)
            percent_reduction = (1 - (len(summary) / len(chunk))) * 100
            print(f'Chunk {count}: reduced by {"%.0f" % percent_reduction}%')

        # Combine all summarized chunks and save to the output file
        final_combined_text = '\n'.join(result)
        save_file(final_combined_text, FILE_OUT)
    except Exception as e:
        logging.critical(f'Fatal error in main application: {e}')
