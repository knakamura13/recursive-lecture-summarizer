import os
import re
import textwrap
import pandas as pd
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


def parse_transcript_file(file_path):
    with open(file_path, 'r') as file:
        data = []
        module_name, video_title, video_transcription = '', '', ''

        for line in file:
            line = line.strip()

            # Check for module name (format: number.number)
            if re.match(r'\d+\.\d+ ', line):
                module_name = line
            # Check for video title (format: number.number.number)
            elif re.match(r'\d+\.\d+\.\d+ ', line):
                # Save previous video details if exist
                if video_title and video_transcription:
                    data.append((module_name, video_title, video_transcription.strip()))
                video_title = line
                video_transcription = ''
            # Accumulate transcription lines
            else:
                video_transcription += line + ' '

        # Adding the last video details
        if video_title and video_transcription:
            data.append([module_name, video_title, video_transcription.strip()])

        # Creating DataFrame
        df = pd.DataFrame(data, columns=['module', 'title', 'transcription'])
        return df


def summarize_lecture_with_gpt(lecture_info, _model="gpt-3.5-turbo", _max_retry=5):
    module, title, transcription = lecture_info

    _messages = [
        {
            "role": "system",
            "content": "You are a writing assistant, skilled in revising and summarizing complex academic and technical "
                       "writing with accuracy and precision."
        },
        {
            "role": "user",
            "content": f"Write a detailed summary of the following text. "
                       f"Preserve as much meaningful information as possible. "
                       f"Treat the summary as if you were writing notes for a graduate-level Computer Science course, "
                       f"with the goal of condensing the information and preserving the most important facts, key terms, "
                       f"and other important information."
                       f"The text is a transcription of a lecture from a Master's level course called CS7641 Machine Learning. \n"
                       f"For your information, this lecture is from a group/module called '{module}'"
                       f"and the lecture is titled '{title}'. "
                       f"This information is only intended to assist you and should not be included in the summary."
                       f"DO NOT respond with anything other than the summary itself."
                       f"\n--BEGIN TEXT TO SUMMARIZE--\n"
                       f"{transcription}"
        }
    ]

    retry = 0
    while retry < _max_retry:
        try:
            response = client.chat.completions.create(model=_model, messages=_messages, timeout=180)
            text = remove_extra_whitespace(response.choices[0].message.content)

            log_text = 'PROMPT:\n\n' + transcription + '\n\n==========\n\nRESPONSE:\n\n' + text
            save_file(log_text, f'gpt3_logs/{time()}_gpt3.txt')

            return text
        except Exception as err:
            retry += 1
            print(f'Retrying ({retry} of {_max_retry})')
            if retry >= _max_retry:
                return "GPT error: %s" % err
            print('Error:', err)
            sleep(1)


if __name__ == '__main__':
    lectures = parse_transcript_file(
        'outputs_archive/cs7641_ml_all_lectures_transcript_raw_text__ch_3_reinforcement_learning.txt')
    result = []

    print(f'Summarizing {lectures.shape[0]} lectures...')
    for idx, row in tqdm(lectures.iterrows(), desc='Summarizing Lectures', leave=True):
        lecture_info = row['module'], row['title'], row['transcription']
        summary = summarize_lecture_with_gpt(lecture_info)
        result.append(f"{lecture_info[0]} - {lecture_info[1]}\n{summary}")

    final_combined_text = '\n\n'.join(result)
    save_file(final_combined_text, 'outputs_archive/gpt3_output_ch_3_reinforcement_learning.txt')
