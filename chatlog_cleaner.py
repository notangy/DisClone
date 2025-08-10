import argparse
import json
from bs4 import BeautifulSoup
import os

# AAAH HORRIBLE HTML KILL IT KILL IT DEAD
# This script will go through every .html file and convert it from hideous markdown
# to nice clean key: value pairs that can be trained with :)
# Key = previous messages in DM or server ('prompt' data)
# Value = response, AKA our own messages

root_folder = './Discord_Export'
output_file = 'training.jsonl'

full_training_data = []


def search_through_logs(soup, discord_username):

    prompt = ''
    response = ''

    # Need a way to track individual conversations to keep training dict sensible
    add_to_data = False
    is_own_msg = False
    prev_own_msg = False
    

    # Messages will only be in <span> elements with the class 'chatlog__markdown-preserve'
    # But we want to ignore any messages that aren't from us
    # Solution: extract spans that are children of the span with title={DISCORD_USERNAME} 

    for div in soup.find_all('div', class_='chatlog__message-group'):
        
        # Extract text from the span that contains the actual message content
        msg = div.find('span', class_='chatlog__markdown-preserve')
        is_own_msg = div.find('span', title=discord_username)

        if msg:
            text = msg.get_text(strip=True)

            # Detect change from own message (True) to not own (False)
            if prev_own_msg and not is_own_msg:
                add_to_data = True  # Conversation flow changed, set flag

            if not is_own_msg:
                # Not own message → prompt
                if not add_to_data:
                    prompt += f"\n {text}"
                else:
                    prompt = text
            else:
                # Own message → response
                response += f"\n {text}"

            # Save current state as previous for next iteration
            prev_own_msg = is_own_msg

            # Append current pair to training data only when add_to_data is True
            if add_to_data:
                full_training_data.append({'prompt': prompt, 'response': response})
                # Optionally reset prompt/response and add_to_data here if starting a new conversation
                prompt = ""
                response = ""
                add_to_data = False

    return full_training_data
        

def read_html(file_path, discord_username):
  # Load your HTML file
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'lxml')

    full_training_data = search_through_logs(soup, discord_username)

    # # Output or save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in full_training_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f'Extracted {file_path} out to {output_file}')
    

def main(discord_username):
    data = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".html"):
                full_path = os.path.join(dirpath, filename)
                print(full_path)
                text = read_html(full_path, discord_username)
                if text:
                    data.append({
                        "text": text,
                        "source_file": os.path.relpath(full_path, root_folder) 
                    })
                else:
                    continue
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clean up discord chat logs for LLM training.')
    parser.add_argument('--username', type=str, help='Your discord username')

    args = parser.parse_args()
    discord_username = args.username

    main(discord_username)