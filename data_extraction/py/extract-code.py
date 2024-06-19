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
    quoting=csv.QUOTE_NONE,
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

dataset_vulnerable = load_dataset(
    "msc-smart-contract-audition/vulnerable-functions-base",
    split="train",
    escapechar="\\",
)

fix_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
The below is a code blocks which might have improper formatting. The task is to fix the formatting and remove comments which do not describe the code. Apart from formatting all of the code needs to remain the same.
1. Ensure that the code is properly indented.
2. Ensure that there are no leading tabs - the outer most block should start with no whitespace infront.
3. If a line is too long break it in multiple lines.
4. Keep all the code the same even if it seems unnecessary in the context.
5. Remove comments which contain no information about the code.
6. Remove any comments which contain URLs.

Do not explain your changes at the end just output the formatted code block.
<|start_header_id|>user<|end_header_id|>
Code block:
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Formatted code block:
```
"""


def extract_codeblocks(description):
    codeblocks = []
    codeblock = ""

    writing = False

    for line in description.split("\n"):
        if writing:
            codeblock += line + "\n"

        if line.strip().startswith("```"):
            if writing:
                codeblocks.append(codeblock)
                yield codeblock
                codeblock = ""
                writing = False
            else:
                writing = True
                codeblock += line + "\n"


# Convert to pandas DataFrame and drop 534 (too long)
descriptions = (
    dataset_vulnerable.to_pandas()
    .drop(index=534)["description"]
    .apply(lambda row: row.replace("\\n", "\n"))
)

chunk_len = 100
current_chunk = 0
start = current_chunk * chunk_len
end = (current_chunk + 1) * chunk_len
print(f"Processing from element {start} to element {end}")
with open(f"cleaned-up-code-{current_chunk}.csv", "w", newline="") as f:
    writer = csv.writer(f, dialect=DIALECT_NAME)

    # Write the header row
    writer.writerow(["code"])

    for description in tqdm(descriptions[start:end], total=chunk_len):
        formatted_blocks = []
        for codeblock in extract_codeblocks(description):
            prompt = fix_template.format(codeblock)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
            output_tokens = model.generate(
                **inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id
            )
            decoded_output = tokenizer.decode(
                output_tokens[0],
                skip_special_tokens=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            formatted_blocks.append(
                decoded_output.split("Formatted code block:\n")[1].strip()
            )

        codeblocks = "\n".join(formatted_blocks).replace("\n", "\\n")
        writer.writerow([codeblocks])
