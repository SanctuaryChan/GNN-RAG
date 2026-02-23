import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from datetime import datetime
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset, load_from_disk
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial

import json

with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}

import re
import string
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def load_qa_dataset(input_path, split):
    if os.path.isdir(input_path) and os.path.exists(
        os.path.join(input_path, "dataset_dict.json")
    ):
        return load_from_disk(input_path)[split]
    return load_dataset(input_path, split=split)


def load_gnn_rag(g_data_file, g_data_file2=None):
    data_file_d = {}
    data_file_gnn = {}

    data_file = os.path.dirname(g_data_file) + "/test.json"
    with open(data_file) as f_in, open(g_data_file) as fg:
        for line, lineg in (zip(f_in, fg)):
            line = json.loads(line)
            lineg = json.loads(lineg)
            
            data_file_d[line["id"]] = line
            data_file_gnn[line["id"]] = lineg
        print("ok1")
    if g_data_file2 is not None:
        data_file = os.path.dirname(g_data_file2) + "/test.json"
        with open(data_file) as f_in, open(g_data_file2) as fg:
            for line, lineg in (zip(f_in, fg)):
                line = json.loads(line)
                lineg = json.loads(lineg)
                
                cand1 = data_file_gnn[line["id"]]["cand"]
                cand2 =  lineg["cand"]

                for c2 in cand2: #c[0] entity c[1] score
                    found=False
                    for c1 in cand1:
                        if c2[0] == c1[0]:
                            if c2[1] > c1[1]: c1[1] = c2[1]
                            found=True
                            break
                    if not found:
                        cand1.append(c2)
                cand1 = sorted(cand1, key=lambda x: x[1], reverse=True)
                data_file_gnn[line["id"]]["cand"] = cand1
            data_file_gnn[line["id"]].update({"cand2": lineg["cand"]})
            print("ok2")

    return data_file_gnn


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def build_llm_input(data, processed_list, input_builder, encrypt=False, data_file_gnn=None):
    question = data["question"]
    answer = data["answer"]
    entities = data['q_entity']
    
    data["cand"] = None
    id = data["id"]
    if data_file_gnn is not None:
        
        lineg = data_file_gnn[data["id"]]
        cand = lineg['cand'] 
        predictiong = []
        for c in cand:
            if c[0] in entities_names:
                predictiong.append(entities_names[c[0]])
            else:
                predictiong.append(c[0])
        data["cand"] = predictiong
    
    
    if id in processed_list:
        return None

    input = input_builder.process_input(data)
    result = {
        "id": id,
        "question": question,
        "ground_truth": answer,
        "input": input,
    }
    return result


def prediction(data, processed_list, input_builder, model, encrypt=False, data_file_gnn=None):
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": data["id"],
            "question": data["question"],
            "prediction": prediction,
            "ground_truth": data["answer"],
            "input": data["question"],
        }

    result = build_llm_input(
        data,
        processed_list=processed_list,
        input_builder=input_builder,
        encrypt=encrypt,
        data_file_gnn=data_file_gnn,
    )
    if result is None:
        return None

    prediction = model.generate_sentence(result["input"])
    if prediction is None:
        return None
    result["prediction"] = prediction.strip()
    return result


def generate_batch_texts(model, prompts):
    if hasattr(model, "generate_batch"):
        return model.generate_batch(prompts)
    return [model.generate_sentence(p) for p in prompts]

def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_qa_dataset(input_file, args.split)
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    data_file_gnn = None
    if os.path.exists(args.rule_path_g1):
        if not os.path.exists(args.rule_path_g2):
            data_file_gnn = load_gnn_rag(args.rule_path_g1)
        else: 
            data_file_gnn = load_gnn_rag(args.rule_path_g1, args.rule_path_g2)



    if not args.strict_answer:
        if args.cot:
            rule_postfix += "_cot"
        if args.explain:
            rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
    if args.strict_answer:
        rule_postfix += "_strict_answer"
        
    print("Load dataset from finished")

    def _safe_filename(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_")

    if args.flat_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "__".join(
            [
                timestamp,
                _safe_filename(args.d),
                _safe_filename(args.model_name),
                _safe_filename(args.split),
                _safe_filename(rule_postfix),
                _safe_filename(str(args.encrypt)),
            ]
        )
        output_dir = os.path.join(args.predict_path, filename)
        output_file = os.path.join(output_dir, "predictions.jsonl")
        args_file = os.path.join(output_dir, "args.txt")
    else:
        output_dir = os.path.join(
            args.predict_path,
            args.d,
            args.model_name,
            args.split,
            rule_postfix,
            str(args.encrypt),
        )
        output_file = os.path.join(output_dir, "predictions.jsonl")
        args_file = os.path.join(output_dir, "args.txt")

    print("Save results to: ", output_file)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.encrypt,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            strict_answer=args.strict_answer,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path,
            args.encrypt,
            args.add_rule,
            use_true=args.use_true,
            strict_answer=args.strict_answer,
        )

    # Save args file
    with open(args_file, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    fout, processed_list = get_output_file(output_file, force=args.force)

    progress = tqdm(total=len(dataset), desc="Predict", dynamic_ncols=True)

    if model is not None and args.batch_size > 1:
        if args.n > 1:
            print("batch_size > 1: disable multiprocessing for inference.")

        def flush_batch(batch_items):
            prompts = [item["input"] for item in batch_items]
            predictions = generate_batch_texts(model, prompts)
            for item, pred in zip(batch_items, predictions):
                if pred is None:
                    continue
                item["prediction"] = pred.strip()
                if args.debug:
                    tqdm.write(json.dumps(item))
                fout.write(json.dumps(item) + "\n")
                fout.flush()

        batch = []
        for data in dataset:
            item = build_llm_input(
                data,
                processed_list=processed_list,
                input_builder=input_builder,
                encrypt=args.encrypt,
                data_file_gnn=data_file_gnn,
            )
            progress.update(1)
            if item is None:
                continue
            batch.append(item)
            if len(batch) >= args.batch_size:
                flush_batch(batch)
                batch = []
        if batch:
            flush_batch(batch)
    elif args.n > 1:
        with Pool(args.n) as p:
            for res in p.imap(
                partial(
                    prediction,
                    processed_list=processed_list,
                    input_builder=input_builder,
                    model=model,
                    encrypt=args.encrypt,
                    data_file_gnn=data_file_gnn

                ),
                dataset,
            ):
                progress.update(1)
                if res is not None:
                    if args.debug:
                        tqdm.write(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in dataset:
            res = prediction(data, processed_list, input_builder, model, encrypt=args.encrypt, data_file_gnn=data_file_gnn)
            progress.update(1)
            if res is not None:
                if args.debug:
                    tqdm.write(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
    progress.close()
    fout.close()

    eval_result(output_file, encrypt=args.encrypt)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="llm backend engine (e.g., vllm); defaults to model_name",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument("--strict_answer", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--rule_path_g1",
        type=str,
        default="results/gnn/webqsp/RoG/test/rearev-sbert/test.info",
    )
    argparser.add_argument(
        "--rule_path_g2",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument(
        "--flat_output",
        action="store_true",
        help="write results into a timestamped single-level directory under predict_path",
    )
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    argparser.add_argument("--encrypt", action="store_true")

    args, _ = argparser.parse_known_args()
    engine_name = args.engine if args.engine is not None else args.model_name
    if engine_name != "no-llm":
        LLM = get_registed_model(engine_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
