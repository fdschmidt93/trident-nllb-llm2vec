# ADAPTED FROM: https://huggingface.co/datasets/juletxara/tydiqa_xtreme/blob/main/tydiqa_xtreme.py
# Why an own dataset? jultxara did not use "goldp" dataset for Korean as per Github page
# Creating own dataset with assertion that context[answer_start + len(answer)] == answer (cf. line 150)

import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

# TODO(tydiqa): BibTeX citation
_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
"""

# TODO(tydiqa):
_DESCRIPTION = """\
TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs.
The languages of TyDi QA are diverse with regard to their typology -- the set of linguistic features that each language
expresses -- such that we expect models performing well on this set to generalize across a large number of the languages
in the world. It contains language phenomena that would not be found in English-only corpora. To provide a realistic
information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but
don’t know the answer yet, (unlike SQuAD and its descendents) and the data is collected directly in each language without
the use of translation (unlike MLQA and XQuAD).

We also include "translate-train" and "translate-test" splits for each non-English languages from XTREME (Hu et al., 2020). These splits are the automatic translations from English to each target language used in the XTREME paper [https://arxiv.org/abs/2003.11080]. The "translate-train" split purposefully ignores the non-English TyDiQA-GoldP training data to simulate the transfer learning scenario where original-language data is not available and system builders must rely on labeled English data plus existing machine translation systems.
"""

_LANG = {
    "ar": "arabic",
    "bn": "bengali",
    "en": "english",
    "fi": "finnish",
    "id": "indonesian",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",
}

_VERSION = datasets.Version("1.1.0", "")


class TyDiQAConfig(datasets.BuilderConfig):
    """BuilderConfig for TydiQa."""

    def __init__(self, lang, **kwargs):
        """

        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(TyDiQAConfig, self).__init__(version=_VERSION, **kwargs)
        self.lang = lang


class TyDiQA(datasets.GeneratorBasedBuilder):
    """TyDi QA: Information-Seeking QA in Typologically Diverse Languages."""

    BUILDER_CONFIGS = [
        TyDiQAConfig(
            name=lang,
            lang=lang,
            description=f"TyDiQA '{lang}' train and test splits, with machine-translated "
            "translate-train/translate-test splits "
            "from XTREME (Hu et al., 2020).",
        )
        for lang in _LANG
        if lang != "en"
    ] + [
        TyDiQAConfig(
            name="en",
            lang="en",
            description="TyDiQA 'en' train and test splits.",
        )
    ]

    def _info(self):
        # TODO(tydiqa): Specifies the datasets.DatasetInfo object

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/google-research-datasets/tydiqa",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(tydiqa): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs

        filepaths = dl_manager.download_and_extract(
            {
                "train": "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json",
                "dev": "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json",
            }
        )

        return [
            datasets.SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": path},
            )
            for split, path in filepaths.items()
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        LANGUAGE = _LANG[self.config.lang]
        with open(filepath, "r") as file:
            data = json.load(file)["data"]
        i = -1
        for article in data:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    id_ = qa["id"]
                    language, _ = id_.split("-", maxsplit=1)
                    if not language == LANGUAGE:
                        continue
                    question = qa["question"]
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]
                    for ans, ans_offset in zip(answers, answer_starts):
                        # CHECK THAT ANSWER IS VALID
                        ans_end = ans_offset + len(ans)
                        assert context[ans_offset:ans_end] == ans
                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    i += 1
                    yield (
                        i,
                        {
                            "id": id_,
                            "context": context,
                            "question": question,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        },
                    )
