# encoding = "utf-8"

import argparse

import json
from scorers.delta import Delta_Scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_name, torch_dtype=torch.bfloat16, device_map="auto",
                                                 use_flash_attention_2=False)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name)
    model.eval()

    print("Testing on {}".format(args.dataset_name))

    '''load dataset'''
    dataset = load_dataset(args.dataset_name, split=args.split).to_pandas()
    source_lines = dataset["lead_with_article"].tolist()
    target_lines = dataset["text"].tolist()

    '''get scores'''
    scorer = Delta_Scorer(model=model, tokenizer=tokenizer, pretrained_name=args.pretrained_name,
                          device=args.device)
    s2s_tok_list, lm_tok_list, prefix_tok_list, s2s_tok_list_doc, lm_tok_list_doc = scorer.compute(sources=source_lines,
                                                                                                   targets=target_lines,
                                                                                                   seperator="TL;DR ")

    '''save to files'''
    model_name = {"LeoLM/leo-mistral-hessianai-7b": "mistral7b"
                  }

    outputpath = "output/" + str(args.dataset_name.replace("/","-")) + "-fflm-" + model_name[
        args.pretrained_name] + f"_{args.split}.jsonl"
    outputfile = open(outputpath, "a+")

    dataset["s2s_tok_list"] = s2s_tok_list
    dataset["lm_tok_list"] = lm_tok_list
    dataset["prefix_tok_list"] = prefix_tok_list
    dataset["s2s_tok_list_1"] = s2s_tok_list_doc
    dataset["lm_tok_list_1"] = lm_tok_list_doc

    dataset.to_json(outputfile, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name",
        type=str,
        default=""
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=""
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    args = parser.parse_args()
    main(args)
