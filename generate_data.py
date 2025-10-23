import sqlite3
import json
import os
import argparse # Keep argparse even if hardcoding username for now
import logging

# --- Configuration ---
DB_NAME = 'instagram_chats.db'
# Corrected prompt file name based on your code
PROMPT_FILE = 'prompt.ini' # Make sure this file exists and contains your prompt
OUTPUT_FILE = 'finetune_data.jsonl'
LOG_FILE = 'export.log' # Log file for this script

# --- Setup Logging for this script ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def read_developer_prompt(filepath=PROMPT_FILE):
    """Reads the developer prompt from the specified file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            if not prompt:
                logging.warning(f"'{filepath}' is empty. Developer prompt will be empty.")
            return prompt
    except FileNotFoundError:
        logging.error(f"Error: '{filepath}' not found. Please create it with your developer prompt.")
        return None
    except Exception as e:
        logging.error(f"Error reading '{filepath}': {e}")
        return None

def get_reviewed_conversations(db_path=DB_NAME):
    """Fetches IDs and titles of reviewed conversations."""
    reviewed_conv_ids = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM conversations WHERE is_reviewed = 1 ORDER BY id")
            rows = cursor.fetchall()
            reviewed_conv_ids = [(row['id'], row['title']) for row in rows]
            logging.info(f"Found {len(reviewed_conv_ids)} reviewed conversations.")
    except sqlite3.Error as e:
        logging.error(f"Database error while fetching reviewed conversations: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching reviewed conversations: {e}")
    return reviewed_conv_ids

def get_messages_for_conversation(db_path, conv_id):
    """Fetches all messages for a specific conversation, ordered by time."""
    messages = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sender_name, content, timestamp_ms FROM messages WHERE conversation_id = ? ORDER BY timestamp_ms ASC",
                (conv_id,)
            )
            messages = cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Database error fetching messages for conversation {conv_id}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching messages for conv {conv_id}: {e}")
    return messages

# --- CORRECTED FUNCTION ---
def format_conversation_for_finetuning(developer_prompt, messages, own_username):
    """Formats a list of messages into the required JSON structure with roles,
       outputting RAW content for assistant messages."""
    if not own_username:
        logging.error("Own username is missing, cannot determine roles.")
        return None

    formatted_messages = [{"role": "developer", "content": developer_prompt}]

    for msg in messages:
        sender = msg['sender_name']
        content = msg['content'] if msg['content'] is not None else "" # Ensure content is a string

        # Determine role based on sender name
        if sender == own_username:
            role = "assistant"
            # --- CHANGE HERE: Output raw content, do NOT add channel structure ---
            formatted_content = content
        else:
            role = "user"
            formatted_content = content # User content remains as is

        # Basic check to avoid empty content messages if desired, though empty strings are handled
        # if not formatted_content.strip() and role != "developer":
        #     logging.debug(f"Skipping potentially empty message from {role} in conv {messages[0].get('conversation_id', 'N/A')}")
        #     continue # Optionally skip messages with only whitespace?

        formatted_messages.append({"role": role, "content": formatted_content})

    # Basic validation: Check if there's at least one user AND one assistant message
    # (The developer prompt doesn't count towards this check)
    has_user = any(m['role'] == 'user' for m in formatted_messages[1:]) # Check messages after developer prompt
    has_assistant = any(m['role'] == 'assistant' for m in formatted_messages[1:])

    if not (has_user and has_assistant):
        logging.warning(f"Skipping conversation: Needs at least one 'user' and one 'assistant' message (excluding developer prompt). Found user={has_user}, assistant={has_assistant}")
        return None # Skip conversations without both roles

    # Final check: Ensure no empty content strings sneak through if that's desired
    # for msg in formatted_messages:
    #     if msg.get("content") is None:
    #          logging.warning(f"Found message with None content for role {msg['role']}, converting to empty string.")
    #          msg["content"] = ""

    return {"messages": formatted_messages}
# --- END CORRECTED FUNCTION ---

def main(your_username):
    """Main function to export data."""
    logging.info("Starting export process...")

    developer_prompt = read_developer_prompt()
    if developer_prompt is None:
        logging.error("Failed to read developer prompt. Exiting.")
        return # Error already logged

    reviewed_conversations = get_reviewed_conversations()
    if not reviewed_conversations:
        logging.info("No reviewed conversations found to export.")
        return

    exported_count = 0
    skipped_count = 0
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            total_conversations = len(reviewed_conversations)
            logging.info(f"Processing {total_conversations} reviewed conversations...")

            for i, (conv_id, conv_title) in enumerate(reviewed_conversations):
                # Reduced logging verbosity inside the loop
                if (i + 1) % 100 == 0:
                     logging.info(f"Processing conversation {i+1}/{total_conversations}...")

                messages = get_messages_for_conversation(DB_NAME, conv_id)

                if not messages:
                    logging.debug(f"Skipping conversation ID {conv_id} ('{conv_title}') as it has no messages.")
                    skipped_count += 1
                    continue

                formatted_data = format_conversation_for_finetuning(developer_prompt, messages, your_username)

                if formatted_data:
                    # Write the formatted conversation as a JSON line
                    try:
                        json_line = json.dumps(formatted_data, ensure_ascii=False)
                        outfile.write(json_line + '\n')
                        exported_count += 1
                    except Exception as json_e:
                        logging.error(f"Error serializing conversation ID {conv_id} to JSON: {json_e}")
                        skipped_count += 1

                else:
                     logging.debug(f"Skipping conversation ID {conv_id} ('{conv_title}') due to formatting issues or missing roles.")
                     skipped_count += 1

        logging.info("-" * 30)
        logging.info(f"Export finished successfully!")
        logging.info(f"Total reviewed conversations found: {total_conversations}")
        logging.info(f"Conversations successfully exported: {exported_count}")
        logging.info(f"Conversations skipped (no messages/roles/errors): {skipped_count}")
        logging.info(f"Output saved to: {OUTPUT_FILE}")
        logging.info("-" * 30)

    except IOError as e:
        logging.error(f"Error writing to output file '{OUTPUT_FILE}': {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during export: {e}", exc_info=True)


if __name__ == "__main__":
    # Using the hardcoded username from your example.
    # For flexibility, using argparse is generally recommended:
    # parser = argparse.ArgumentParser(...)
    # parser.add_argument("your_username", ...)
    # args = parser.parse_args()
    # your_username = args.your_username
    your_username = "درمانگر مشکلات جنسی | دکتر محمدی" # Hardcoded username

    if not your_username:
        print("Error: The Instagram username is empty. Please provide it either via command line argument or by editing the script.")
        exit(1)

    print(f"Using username: '{your_username}' to identify assistant messages.")
    main(your_username)