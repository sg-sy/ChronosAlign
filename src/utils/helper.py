import json
import pandas as pd


def extract_rephrase_from_jsonl(file_path):
    """
    Extract rephrased question text from an Anthropic JSONL batch results file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        pd.DataFrame: DataFrame with columns `id` and `rephrased_question`.
    """
    messages = []
    error_counter = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                custom_id = data.get("custom_id")

                text_content = ""
                if "result" in data and "message" in data["result"]:
                    message = data["result"]["message"]
                    if "content" in message and len(message["content"]) > 0:
                        text_content = message["content"][0].get("text", "")

                cleaned = text_content.strip()

                # Handle occasional fenced response formatting.
                if cleaned.startswith("```"):
                    parts = cleaned.split("```")
                    if len(parts) >= 2:
                        cleaned = parts[1].strip()
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:].strip()

                # Prefer JSON payload if model returned {"rephrased_question": "..."}.
                rephrased_question = cleaned
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "rephrased_question" in parsed:
                        rephrased_question = str(parsed["rephrased_question"]).strip()
                    elif isinstance(parsed, str):
                        rephrased_question = parsed.strip()
                except json.JSONDecodeError:
                    pass

                messages.append(
                    {
                        "id": str(custom_id) if custom_id is not None else None,
                        "rephrased_question": rephrased_question,
                    }
                )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_counter += 1
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                error_counter += 1
                continue

    print(f"Total errors: {error_counter}")
    return pd.DataFrame(messages)


def extract_messages_from_jsonl(file_path):
    """
    Extract messages from JSONL file and create a DataFrame with id column.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        pd.DataFrame: DataFrame with extracted messages and id column
    """
    messages = []
    error_counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Extract custom_id as the id
                custom_id = data.get("custom_id")

                # Extract message content
                if "result" in data and "message" in data["result"]:
                    message = data["result"]["message"]

                    # Get the text content from the message
                    if "content" in message and len(message["content"]) > 0:
                        text_content = message["content"][0].get("text", "")

                        # Try to parse the JSON content within the text
                        try:
                            # Extract JSON from the text (remove markdown code blocks if present)
                            json_text = text_content.strip()

                            # Remove markdown code block markers
                            parts = json_text.split("```")
                            if len(parts) >= 2:
                                json_text = parts[1]
                                if json_text.startswith("json"):
                                    json_text = json_text[4:]

                            # Clean up the JSON text
                            json_text = json_text.strip()

                            # Parse the JSON content
                            temporal_data = json.loads(json_text)

                            # Add id to each temporal data entry
                            for entry in temporal_data:
                                entry["id"] = custom_id
                                messages.append(entry)

                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: Could not parse JSON content in line {line_num}: {e}"
                            )
                            error_counter += 1
                            # If JSON parsing fails, add the raw text with id
                            messages.append(
                                {
                                    "id": custom_id,
                                    "raw_content": (
                                        text_content[:200] + "..."
                                        if len(text_content) > 200
                                        else text_content
                                    ),
                                    "parse_error": str(e),
                                }
                            )
                    else:
                        # No content found, add basic message info
                        messages.append(
                            {
                                "id": custom_id,
                                "message_id": message.get("id"),
                                "role": message.get("role"),
                                "model": message.get("model"),
                                "content_available": False,
                            }
                        )
                else:
                    # No message found, add basic data info
                    messages.append(
                        {
                            "id": custom_id,
                            "result_type": data.get("result", {}).get("type"),
                            "message_available": False,
                        }
                    )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    print(f"Total errors: {error_counter}")
    return pd.DataFrame(messages)


def extract_simple_message_from_jsonl(file_path, column_name):
    """
    Extract messages from JSONL file and create a DataFrame with id column.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        pd.DataFrame: DataFrame with extracted messages and id column
    """
    messages = []
    error_counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Extract custom_id as the id
                custom_id = data.get("custom_id")

                # Extract message content
                if "result" in data and "message" in data["result"]:
                    message = data["result"]["message"]

                    # Get the text content from the message
                    if "content" in message and len(message["content"]) > 0:
                        text_content = message["content"][0].get("text", "")

                        messages.append({"id": custom_id, column_name: text_content})

                    else:
                        # No content found, add basic message info
                        messages.append(
                            {
                                "id": custom_id,
                                "message_id": message.get("id"),
                                "role": message.get("role"),
                                "model": message.get("model"),
                                "content_available": False,
                            }
                        )
                else:
                    # No message found, add basic data info
                    messages.append(
                        {
                            "id": custom_id,
                            "result_type": data.get("result", {}).get("type"),
                            "message_available": False,
                        }
                    )

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    print(f"Total errors: {error_counter}")
    return pd.DataFrame(messages)
