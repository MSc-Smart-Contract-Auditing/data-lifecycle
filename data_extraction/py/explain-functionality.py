from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import csv
import sys

if len(sys.argv) != 2:
    print("Usage: python my_script.py <current_chunk>")
    sys.exit(1)

current_chunk = int(sys.argv[1])
print(f"Executing chunk: {current_chunk}")
DIALECT_NAME = "db_dialect"

csv.register_dialect(
    DIALECT_NAME,
    delimiter=",",
    quoting=csv.QUOTE_MINIMAL,
    escapechar="\\",
)

max_seq_length = 20  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

codeblocks = pd.read_csv("~/cleaned-up-code.csv")

template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below are provided 1 or more code blocks from a smart contract. The task is to explain what is its purpose in plain English.

1. Do NOT use any names of variables, names of parameters or names of functions. Just try to explain the general functionality.
2. Explain what each codeblock does step by step.
3. Give a high-level overview and purpose of the code.
4. First explain the functionality of each code block and after all codeblocks are explained provide a high-level overview of the code over all.

<|start_header_id|>user<|end_header_id|>
Code block to explain:
{}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Code functionality:
Code block 1:
1. """

chunk_len = 100
start = current_chunk * chunk_len
end = (current_chunk + 1) * chunk_len

with open(f"functionality-{current_chunk}.csv", "w", newline="") as f:
    writer = csv.writer(f, dialect=DIALECT_NAME)

    # Write the header row
    writer.writerow(["functionality"])

    for codeblock in tqdm(codeblocks["code"][start:end], total=chunk_len):

        print(codeblock.replace("\\n", "\n"))

        prompt = template.format(codeblock.replace("\\n", "\n"))

        input = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

        output_tokens = model.generate(
            **input, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id
        )

        decoded_output_purpose = tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        functionality = decoded_output_purpose.split("Code functionality:\n")[1].strip()
        writer.writerow([functionality.replace("\n", "\\n")])
