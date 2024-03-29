import inquirer
import re

import json
import textwrap

from langchain.docstore.document import Document
from datetime import datetime, timedelta

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")

import os.path

def updateLogs(logs, getUserResponse) : 
    if (getUserResponse) :
        logsName = "userLogs.json"
    else : 
        logsName = "testLogs.json"

    # Save logs 
    if os.path.isfile(logsName) :
            with open(logsName, 'r') as fp:
                allLogs = json.load(fp)
    else : 
        allLogs = []    

    allLogs += logs
    with open(logsName, 'w') as fp:
        json.dump(allLogs, fp)    

def countTokens(text) : 
    tokens = tokenizer(text)
    return len(tokens['input_ids'])


def wrap_text_preserve_newlines(text, width=130):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def split_by_number(s):
    # Pattern to split by number followed by a period and space, keeping the number
    pattern = r'(?<=\d\.)\s'
    # Splitting the string
    parts = re.split(pattern, s)
    # Removing the first empty element if present
    if parts and parts[0] == '':
        parts = parts[1:]
    return parts

def dropdown_selection(options):
    questions = [
        inquirer.List('selection',
                      message="Select an option",
                      choices=options,
                     ),
    ]
    answers = inquirer.prompt(questions)
    return answers['selection']


def convert_to_normalized_sentence(multiline_string):
    words = multiline_string.split()
    return ' '.join(words)


def createDoc(out, char, name) : 
    doc = Document(page_content=out)
    #doc.page_content = out

    doc.metadata = {
        'source'  : 'interaction',
        'type' : 'public'
    }

    datetime_obj = datetime.now()

    doc.metadata['char'] = char
    doc.metadata['name'] = name
    doc.metadata['time'] = datetime_obj.strftime("%Y-%m-%dT%H:%M:%S")

    return doc




def timedelta_to_string(td):
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds // 60) % 60
    seconds = td.seconds - (hours * 3600) - (minutes * 60)

    if days > 0:
        return f"{days} days"
    elif hours > 0:
        return f"{hours} hours"
    elif minutes > 0:
        return f"{minutes} minutes"
    elif seconds > 0:
        return f"{seconds} seconds"
    else:
        return "0 seconds"