import os
import re
import textwrap
from tqdm import tqdm
from openai import OpenAI
from time import time, sleep
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('API_KEY'))


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text.strip())


def summarize_with_gpt(_text, _model="gpt-3.5-turbo", _max_retry=5):
    _messages = [
        {
            "role": "system",
            "content": "You are a writing assistant, skilled in revising and summarizing complex technical writing with accuracy and "
                       "precision."
        },
        # {
        #     "role": "user",
        #     "content": f"Write a concise summary of the following text. "
        #                f"Use abbreviations to shorten words and phrases wherever possible. For example, never use the phrases 'Machine "
        #                f"Learning', 'Reinforcement Learning', or 'Unsupervised Learning', because you could instead use their "
        #                f"abbreviations 'ML', 'RL', and 'UL'. You can also abbreviate individual words "
        #                f"(e.g., 'alg' instead of 'algorithm' and 'avg' instead of 'average'). "
        #                f"DO NOT respond with anything other than the summary itself. "
        #                f"--BEGIN TEXT TO SUMMARIZE-- "
        #                f" {_text}"
        # }
        # {
        #     "role": "user",
        #     "content": f"Summarize this text, preserving as much detail as possible. You should prioritize accuracy and preserving "
        #                f"information, with your secondary goal being a reduction in word count. Focus on removing redundant information."
        #                f"Do not include anything in your response except for the summarized text. "
        #                f"\n--BEGIN TEXT TO SUMMARIZE-- \n"
        #                f" {_text}"
        # }
        {
            "role": "user",
            "content": f"Write a concise summary of the following text. "
                       f"DO NOT respond with anything other than the summary itself. "
                       f"--BEGIN TEXT TO SUMMARIZE-- "
                       f" {_text}"
        }
    ]

    retry = 0
    while retry < _max_retry:
        try:
            response = client.chat.completions.create(model=_model, messages=_messages, timeout=180)
            text = remove_extra_whitespace(response.choices[0].message.content)

            log_text = 'PROMPT:\n\n' + _text + '\n\n==========\n\nRESPONSE:\n\n' + text
            save_file(log_text, f'gpt4_logs/{time()}_gpt3.txt')

            return text
        except Exception as err:
            retry += 1
            print(f'Retrying ({retry} of {_max_retry})')
            if retry >= _max_retry:
                return "GPT error: %s" % err
            print('Error:', err)
            sleep(1)


if __name__ == '__main__':
    original_text = open_file('input.txt')
    chunks = textwrap.wrap(original_text, 1000)
    num_chunks = len(chunks)

    result = []
    count = 0

    for chunk in tqdm(chunks, desc='Summarizing chunks', leave=True):
        count += 1
        summary = summarize_with_gpt(chunk)
        result.append(summary)

        len_old = len(chunk)
        len_new = len(summary)
        percent_reduction = (1 - (len_new / len_old)) * 100
        print(f'{count}: reduction = {"%.0f" % percent_reduction}%')

    final_combined_text = '\n'.join(result)
    save_file(final_combined_text, 'output.txt')
