import sys

from chat_completions_fine_tuned import chat_completion_fine_tuned
from data_set import DataSet
from fine_tuning import FineTuning
import argparse


def main():
    parser = argparse.ArgumentParser(description='Bypass fine-tuning and go directly to chat completion')
    parser.add_argument('--bypass_fine_tuning', action='store_true', help='Bypass fine-tuning')
    parser.add_argument('--model_id', type=str, required='--bypass_fine_tuning' in sys.argv, help='Model ID')
    parser.add_argument('--question', type=str, help='Question string')
    args = parser.parse_args()

    if args.bypass_fine_tuning:
        print(args.model_id)
        chat_completion_fine_tuned(args.model_id, question=args.question)
    else:
        print(f"\n {'*' * 20 } Starting Fine-Tuning... {'*' * 20 }")
        ds = DataSet()
        print(f"\n {'*' * 20 } Validating Data Set... {'*' * 20 }")
        ds.validate_data_set()
        ft = FineTuning(data_file="data.jsonl")
        print(f"\n {'*' * 20 } Creating File... {'*' * 20 }")
        ft.create_file()
        print(f"\n {'*' * 20 } Creating Job... {'*' * 20 }")
        ft.create_job()
        print(f"\n {'*' * 20 } Getting Model ID... {'*' * 20 }")
        ft.get_model_id()

        chat_completion_fine_tuned(ft.fine_tuned_model_id, question=args.question)


if __name__ == '__main__':
    main()
