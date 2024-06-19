from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import csv

csv_params = {
    "quoting": csv.QUOTE_MINIMAL,  # quoting
    "quotechar": '"',  # quotechar
    "escapechar": "\\",  # escapechar
    "doublequote": False,  # Ensure doublequote is set to False as you are using escapechar
}


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
The below vulnerability audit contains a description of a vulnerability contained in the code blocks.
All the text needs to be extracted removing just the codeblocks. The inline code should be kept as is. The text needs to remain the same but you may fix any typos or grammatical errors.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Vulnerability audit:
{}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Vulnerability text:
"""


# Convert to pandas DataFrame
df_vulnerable = dataset_vulnerable.to_pandas()
queries = df_vulnerable["description"].apply(
    lambda x: query_template.format(x.replace("\\n", "\n"))
)
queries = queries.drop(queries.index[534])


classes = []
for query in tqdm(queries):
    inputs = tokenizer(query, return_tensors="pt", truncation=True).to("cuda")
    output_tokens = model.generate(
        **inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id
    )
    decoded_output = tokenizer.decode(
        output_tokens[0], skip_special_tokens=True, pad_token_id=tokenizer.pad_token_id
    )
    classes.append(decoded_output.split("Vulnerability type:\n")[1].strip().lower())

df = pd.DataFrame(classes, columns=["description"])
df.to_csv("vulnerability_descriptions.csv", index=False, **csv_params)
