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

query_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
The below vulnerability audit contains an audit name, a description of a vulnerability and a mitigation.
You need to improve the mitigation based on the information provided.
1. Expand the mitigation in such a way that it is very comprehensive and easy to understand.
2. Do not dumb it down, still use technical terms where necessary.
3. Do not include the code blocks, but you could still use inline code for your explanation. Surround the inline code using single backticks.
4. In your reponse do not quote the title of the vulnerability, add anything about the description or the code. Just output the enhanced mitigation
5. Do not make up any information about the vulnerability that is not in the provided context.
<|start_header_id|>user<|end_header_id|>
Vulnerability title: {}
Vulnerability description:
{}
Vulnerability mitigation:
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Improved mitigation:
"""

query_template_empty = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
The below vulnerability audit contains an audit name and description of the vulnerability.
Write one suggestion for the mitigation of the vulnerability.
1. Expand the mitigation in such a way that it is very comprehensive and easy to understand.
2. Do not dumb it down, still use technical terms where necessary.
3. Do not include the code blocks, but you could still use inline code for your explanation. Surround the inline code using single backticks.
4. In your reponse do not quote the title of the vulnerability, add anything about the description or the code. Just output the enhanced mitigation
5. Do not make up any information about the vulnerability that is not in the provided context.
<|start_header_id|>user<|end_header_id|>
Vulnerability title: {}
Vulnerability description:
{}
<|start_header_id|>assistant<|end_header_id|>
Improved mitigation:
"""


def apply_template(row):
    if row["recommendation"] is None:
        return query_template_empty.format(
            row["name"],
            row["description"].replace("\\n", "\n"),
        )

    return query_template.format(
        row["name"],
        row["description"].replace("\\n", "\n"),
        row["recommendation"].replace("\\n", "\n"),
    )


# Convert to pandas DataFrame
df_vulnerable = dataset_vulnerable.to_pandas()
queries = df_vulnerable.apply(apply_template, axis=1)
queries = queries.drop(queries.index[534])

chunk_len = 100
start = current_chunk * chunk_len
end = (current_chunk + 1) * chunk_len

with open(f"enhanced-recommendation-{current_chunk}.csv", "w", newline="") as f:
    writer = csv.writer(f, dialect=DIALECT_NAME)

    # Write the header row
    writer.writerow(["recommendation"])

    for query in tqdm(queries[start:end], total=chunk_len):
        inputs = tokenizer(query, return_tensors="pt", truncation=True).to("cuda")
        output_tokens = model.generate(
            **inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id
        )
        decoded_output = tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        mitigation = (
            decoded_output.split("Improved mitigation:\n")[1]
            .strip()
            .replace("\n", "\\n")
        )
        writer.writerow([mitigation])
