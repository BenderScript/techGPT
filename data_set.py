# This is OpenAI's code with the following changes:
# 1 - Created functions for each validation step
# 2 - Mixed in some Azure AI code since they have fantastic examples

import json
import tiktoken  # for token counting
import numpy as np
from collections import defaultdict


class DataSet:

    def __init__(self, encoding=None, data_path="data.jsonl"):

        if encoding is None:
            self.encoding = tiktoken.get_encoding(
                "cl100k_base")  # default encoding used by gpt-4, turbo, and text-embedding-ada-002 models

        self.total_tokens = []
        self.assistant_tokens = []
        self.data_path = data_path

    def num_tokens_from_messages(self, messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(self.encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(self, values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    def check_dataset(self, training_data):
        # Initial dataset stats
        print("Num examples:", len(training_data))
        print("First example:")
        for message in training_data[0]["messages"]:
            print(message)

        # Format error checks
        format_errors = defaultdict(int)

        for ex in training_data:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name", "function_call") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant", "function"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

            self.total_tokens.append(self.num_tokens_from_messages(messages))
            self.assistant_tokens.append(self.num_assistant_tokens_from_messages(messages))

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
            return False
        else:
            print("No errors found")
            return True

    def compute_pricing(self, training_data):
        # Pricing and default n_epochs estimate
        MAX_TOKENS_PER_EXAMPLE = 4096
        TARGET_EPOCHS = 3
        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        MIN_DEFAULT_EPOCHS = 1
        MAX_DEFAULT_EPOCHS = 25

        n_epochs = TARGET_EPOCHS
        n_train_examples = len(training_data)
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

        n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in self.total_tokens)
        print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

    # Load the dataset
    def validate_data_set(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
                if self.check_dataset(dataset) is False:
                    return False
                self.print_distribution(self.total_tokens, "total tokens")
                self.print_distribution(self.assistant_tokens, "assistant tokens")
                print('*' * 50)
                self.compute_pricing(dataset)
                return True
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return False
        except Exception as e:
            print(f"Unknown Error: {e}")
            return False
