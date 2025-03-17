import json
import time

import fire
import numpy as np
import vllm
from datasets import load_from_disk

from simple_math_eval.grader import math_reward_fn


def apply_qwen_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    template: str = "qwen",
    dataset_name: str = "./test_data",
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    max_test: int = 999999,
):

    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=2,
        seed=int(time.time_ns()),
    )

    model = vllm.LLM(model_name, swap_space=32, max_model_len=max_model_len)

    if template == "qwen":
        print("use apply_qwen_math_template")
        apply_template = apply_qwen_template
    else:
        raise ValueError

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    to_be_saved = []
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue
        prompts = dataset["problem"][:max_test]
        targets = dataset["answer"][:max_test]

        prompts = list(map(apply_template, prompts))
        print("inference for ", task_name)
        outputs = model.generate(prompts, sampling_params)
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        for k in range(len(outputs)):
            output = outputs[k]
            gt = [targets[k]] * sampling_params.n
            rewards = []
            infos = []
            for o in output.outputs:
                info, reward = math_reward_fn(o.text, gt)
                rewards.append(reward)
                infos.append(info)
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            if infos[0] is not {}:
                batch_formatted.append(np.array([i["formatted"] for i in infos]).sum())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": gt,
                    "model_output": [o.text for o in output.outputs],
                    "reward": [r for r in rewards],
                }
            )

        results[task_name] = np.mean(batch_scores)
        avg_lens[task_name] = np.mean(batch_lengths)
        if batch_formatted:
            formatted[task_name] = np.mean(batch_formatted)
        max_lens[task_name] = np.max(batch_lengths)

    print(results)
    print("avg:", np.mean(list(results.values())))
    print("avg_lens:", avg_lens)
    print("max_lens:", max_lens)
    print("formatted:", formatted)

    fn = "model_eval_out_" + model_name.replace("/", "_") + str(int(time.time()))
    json.dump(to_be_saved, open(f"{fn}_template_{template}.json", "w"), indent=4)


fire.Fire(main)
