import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

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

class PromptBuilder(object):
    MCQ_INSTRUCTION = """Please answer the following questions. Select from the given choices and return only the chosen answer(s). No explanation, no extra text. One answer per line."""
    SAQ_INSTRUCTION = """Please answer the following questions. Return all possible answers only, with one answer per line. No explanation, no extra text, no numbering."""
    MCQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Select from the given choices and return only the chosen answer(s). Use the final entity of each reasoning path and output all distinct answers. No explanation, no extra text. One answer per line."""
    SAQ_RULE_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Use the final entity of each reasoning path and output all distinct answers. Return all answers only, with one answer per line. No explanation, no extra text, no numbering."""
    #SAQ_RULE_INSTRUCTION = """Based on the provided knowledge, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
    #SAQ_RULE_INSTRUCTION = """Your tasks is to use the following facts and answer the question. Make sure that you use the information from the facts provided. Please keep the answer as simple as possible and return all the possible answers as a list."""
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    #GRAPH_CONTEXT = """The facts are the following:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(
        self,
        prompt_path,
        encrypt=False,
        add_rule=False,
        use_true=False,
        cot=False,
        explain=False,
        use_random=False,
        each_line=False,
        maximun_token=4096,
        tokenize: Callable = lambda x: len(x),
        use_verbalizer=False,
        verbalizer_mode="plain",
        verbalizer_operator="auto",
        reranker=None,
        reranker_topk=0,
    ):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line

        self.encrypt=encrypt
        self.use_verbalizer = use_verbalizer
        self.verbalizer_mode = verbalizer_mode
        self.verbalizer_operator = verbalizer_operator
        self.reranker = reranker
        self.reranker_topk = reranker_topk
        # If reranker is used, preserve ordering when trimming for length.
        self.preserve_order = reranker is not None

    def _format_path_text(self, path, question):
        if not self.use_verbalizer:
            return utils.path_to_string(path)
        operator = None if self.verbalizer_operator == "auto" else self.verbalizer_operator
        answer = None
        if self.verbalizer_mode == "answer" and path:
            answer = path[-1][-1]
        entities_names = utils.get_entities_names()
        text = utils.verbalize_path(
            path,
            question=question,
            operator=operator,
            entity_map=entities_names,
            answer=answer,
        )
        if text is None or text.strip() == "":
            return utils.path_to_string(path)
        return text
        
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        
        entities = question_dict['q_entity']
        skip_ents = []
        
        graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)

        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        
        lists_of_paths = []
        lists_of_answers = []
        lists_of_rel_paths = []
        if self.add_rule:
            entities = question_dict['q_entity']
            #graph = utils.build_graph(question_dict['graph'], entities, self.encrypt)
            skip_ents = []
            
            graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = []
                lists_of_answers = []
                lists_of_rel_paths = []
                for p in reasoning_paths:
                    lists_of_paths.append(self._format_path_text(p, question))
                    lists_of_answers.append(p[-1][-1] if p else None)
                    lists_of_rel_paths.append([r for _, r, _ in p] if p else [])
                
                # context = "\n".join([utils.path_to_string(p) for p in reasoning_paths])
            else:
                lists_of_paths = []
            #input += self.GRAPH_CONTEXT.format(context = context)
        #lists_of_paths = []
        if question_dict['cand'] is not None:
            if not self.add_rule:
                skip_ents = []
                graph = utils.build_graph(question_dict['graph'], skip_ents, self.encrypt)
            lists_of_paths2 = []
            #print(question_dict['cand'])
            reasoning_paths = utils.get_truth_paths(question_dict['q_entity'], question_dict['cand'], graph)
            for p in reasoning_paths:
                p_text = self._format_path_text(p, question)
                if p_text not in lists_of_paths:
                    lists_of_paths.append(p_text)
                    lists_of_answers.append(p[-1][-1] if p else None)
                    lists_of_rel_paths.append([r for _, r, _ in p] if p else [])
            
            for p in reasoning_paths:
                p_text = self._format_path_text(p, question)
                if p_text not in lists_of_paths2:
                    lists_of_paths2.append(p_text)
           
        input = self.QUESTION.format(question = question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_rule or question_dict['cand'] is not None:
                instruction = self.MCQ_RULE_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_rule or question_dict['cand'] is not None:
                instruction = self.SAQ_RULE_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_rule or question_dict['cand'] is not None:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            # optional reranking before prompt truncation
            if self.reranker is not None and self.reranker_topk and lists_of_paths:
                lists_of_paths = self.reranker.rerank(
                    question=question,
                    path_texts=lists_of_paths,
                    answers=lists_of_answers,
                    rel_paths=lists_of_rel_paths,
                    topk=self.reranker_topk,
                )
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            
            input = self.GRAPH_CONTEXT.format(context = context) + input
        
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, trim paths.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # If reranker is used, preserve path order (higher score first).
            if not self.preserve_order:
                random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
            
