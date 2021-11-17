import datasets
import typing as tp
from pathlib import Path
import csv

_DESCRIPTION = ""  # TODO

_HOMEPAGE = "https://github.com/lksenel/Kardes-NLU/"  # TODO

_LICENSE = "https://opendatacommons.org/licenses/by/1-0/"

_URL = "https://raw.githubusercontent.com/lksenel/Kardes-NLU/main/Data/{lang}/{task}.{split}.{lang_code}.csv"

GoldLabel2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
LANG_CODE2LANG = {
    "az": "azeri",
    "kk": "kazakh",
    "ky": "kyrgyz",
    "ug": "uyghur",
    "uz": "uzbek",
}

TASKS = ["xnli", "sts", "xcopa"]


class KardesNLUConfig(datasets.BuilderConfig):
    """BuilderConfig for KardesNLU Dataset."""

    def __init__(self, task_lang, **kwargs):
        super(KardesNLUConfig, self).__init__(**kwargs)
        self.task_lang = task_lang
        self.task, self.lang = self.task_lang.split(".")


def _builder_configs() -> tp.List[KardesNLUConfig]:
    configs = []
    for lang_code in LANG_CODE2LANG.keys():
        for task in TASKS:
            cfg = KardesNLUConfig(
                name=f"{task}.{lang_code}",
                version=datasets.Version("1.0.0"),
                description=f"KardesNLU: {task} - {lang_code}",
                task_lang=f"{task}.{lang_code}",
            )
            configs.append(cfg)
    return configs


class KardesNLU(datasets.GeneratorBasedBuilder):
    """KardesNLU Dataset."""

    BUILDER_CONFIGS = _builder_configs()
    BUILDER_CONFIG_CLASS = KardesNLUConfig

    def _info(self):
        features = datasets.Features(
            {
                "label": datasets.ClassLabel(
                    names=["entailment", "neutral", "contradiction"]
                ),
                "premise": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "premise_english": datasets.Value("string"),
                "hypothesis_english": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        if not self.config.name == "all":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "file_path": dl_manager.download_and_extract(
                            _URL.format(
                                lang_code=self.config.lang,
                                lang=LANG_CODE2LANG[self.config.lang],
                                task=self.config.task,
                                split="dev",
                            )
                        )
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "file_path": dl_manager.download(
                            _URL.format(
                                lang_code=self.config.lang,
                                lang=LANG_CODE2LANG[self.config.lang],
                                task=self.config.task,
                                split="test",
                            )
                        )
                    },
                ),
            ]

    def _generate_examples(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for id_, row in enumerate(reader):
                yield (
                    id_,
                    {
                        "label": GoldLabel2ID[row["gold_label"]],
                        "premise_english": row["sentence1"],
                        "hypothesis_english": row["sentence2"],
                        "premise": row["s1_translation"],
                        "hypothesis": row["s2_translation"],
                    },
                )

