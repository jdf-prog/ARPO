import datasets
import json
dataset_names = ["webwalker", "xbench", "bamboogle", "2wiki", "hotpotqa", "musique", "SimpleQA", "hle", "gaia"]
prompt_type = ["code_search"] * len(dataset_names)
prompt_type[0:2] = ["code_search_cn", "code_search_cn"]  # webwalker and xbench use Chinese prompts

code_search_prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

code_search_cn_prompt = """你是一个乐于助人的助手，能够借助 Wikipedia 搜索工具和 Python 解释器工具，逐步解决给定的问题。
给定一个问题后，你需要先在头脑中进行推理过程，然后再提供答案。
在思考过程中，你可以调用 Wikipedia 搜索工具来搜索特定主题的事实信息，也可以使用 Python 解释器工具来计算数学问题（如有需要）。
推理过程和答案分别用 <think> 和 </think>，以及 <answer> 和 </answer> 标签括起来；
搜索查询和结果分别用 <search> 和 </search>，以及 <result> 和 </result> 标签括起来。
例如：
<think> 这是推理过程。 </think> <search> 这里是搜索查询 </search> <result> 这里是搜索结果 </result>
<think> 这是推理过程。 </think> <python> 这里是 Python 代码 </python> <result> 这里是 Python 解释器的结果 </result>
<think> 这是推理过程。 </think> <answer> 最终答案是 \\[ \\boxed{这里是答案} \\] </answer>"""


train_dataset = datasets.load_dataset('parquet', data_files="train.parquet")['train']

def process_train_item(item, index):
    answer = item['reward_model']['ground_truth']
    return {
        "data_source": item['data_source'],
        "prompt": item['prompt'],
        "ability": "code_search",
        "reward_model": {
            "ground_truth": [answer] if isinstance(answer, str) else answer,
        },
        "extra_info": item['extra_info'],
        "metadata": ""
    }
processed_train_dataset = train_dataset.map(
    process_train_item,
    with_indices=True,
    remove_columns=train_dataset.column_names
)

from pathlib import Path
data_dir = Path("evaluation/data")
dataset_dict = {}
for dataset_name, prompt_type in zip(dataset_names, prompt_type):
    system_prompt = code_search_prompt if prompt_type == "code_search" else code_search_cn_prompt
    data_file = data_dir / dataset_name / "test.jsonl"
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    dataset = datasets.Dataset.from_list(data)
    
    def process_item(item, index):
        metadata = {k: v for k, v in item.items() if k not in ['question', 'answer']}
        metadata = json.dumps(metadata, ensure_ascii=False)
        if 'question' in item:
            question = item['question']
        elif 'Question' in item:
            question = item['Question']
        else:
            raise KeyError("Neither 'question' nor 'Question' found in item, item.keys() = {}".format(item.keys()))
        answer = item['answer']
        return {
            "data_source": dataset_name,
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "ability": "code_search",
            "reward_model": {
                "ground_truth": [answer] if isinstance(answer, str) else answer,
            },
            "extra_info": {
                "index": index,
                "question": question,
                "split": "test",
            },
            "metadata": metadata,
        }
    processed_dataset = dataset.map(
        process_item,
        with_indices=True,
        remove_columns=dataset.column_names
    )
    # processed_dataset.push_to_hub(
    #     "VerlTool/deepsearch", split=f"test_{dataset_name}",
    # )
    dataset_dict[f"test_{dataset_name}"] = processed_dataset
    print(f"Processed {dataset_name} dataset and pushed to VerlTool/deepsearch with split test_{dataset_name}.")    
    


dataset_dict["train"] = processed_train_dataset
dataset_dict = datasets.DatasetDict(dataset_dict)
dataset_dict.push_to_hub(
    "VerlTool/deepsearch", 
)