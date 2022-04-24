"""
KLUE
https://arxiv.org/abs/2105.09680

 Korean Language Understanding Evaluation (KLUE) benchmark is a series of datasets
 to evaluate natural language understanding capability of Korean language models.
 KLUE consists of 8 diverse and representative tasks, which are accessible to anyone without any restrictions.
 With ethical considerations in mind, we deliberately design annotation guidelines
 to obtain unambiguous annotations for all datasets. Furthermore, we build an evaluation system
 and carefully choose evaluations metrics for every task, thus establishing fair comparison across Korean language models.
 
 Homepage: https://klue-benchmark.com/
"""

import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import macro_f1_score


class YNAT(Task): # MultipleChoiceTask로 변환 필요
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "ynat"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "다음 문장의 카테고리는?\n{}\n답변:".format(doc["title"])

    def doc_to_target(self, doc):
        return " {}".format({0: "과학", 1: "경제", 2: "사회", 3: "생활", 4: "세계", 5: "스포츠", 6: "정치"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_sci, _ = rf.loglikelihood(ctx, " 과학")
        ll_eco, _ = rf.loglikelihood(ctx, " 경제")
        ll_soc, _ = rf.loglikelihood(ctx, " 사회")
        ll_life, _ = rf.loglikelihood(ctx, " 생활")
        ll_wor, _ = rf.loglikelihood(ctx, " 세계")
        ll_spo, _ = rf.loglikelihood(ctx, " 스포츠")
        ll_pol, _ = rf.loglikelihood(ctx, " 정치")
        return ll_sci, ll_eco, ll_soc, ll_life, ll_wor, ll_spo, ll_pol

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["label"]
        return {
            "f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "f1": True
        }

    def aggregation(self):
        return {
            "f1": macro_f1_score
        }
