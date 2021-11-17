# The way https://huggingface.co/datasets/facebook/belebele is uploaded makes loading data per language extremely slow
# since, for every language, the entire dataset is downloaded as languages are treated as splits and not separate sub datasets

import datasets
import typing as tp

_DESCRIPTION = ""  # TODO

LANGS = [
    # "aym",
    "bzd",
    "cni",
    # "gn",
    "hch",
    "nah",
    "oto",
    # "quy",
    "shp",
    "tar",
]

lang_code2lang = {
    "cni": "ashaninka",
    # "aym": "aymara",
    "bzd": "bribri",
    # "gn": "guarani",
    "oto": "hñähñu",
    "nah": "nahuatl",
    # "quy": "quechua",
    "tar": "raramuri",
    "shp": "shipibo_konibo",
    "hch": "wixarika",
}


class AmericasNLPConfig(datasets.BuilderConfig):
    """BuilderConfig for KardesNLU Dataset."""

    def __init__(self, name, **kwargs):
        super(AmericasNLPConfig, self).__init__(**kwargs)
        self.name = name


def _builder_configs() -> tp.List[AmericasNLPConfig]:
    configs = []
    for lang in LANGS:
        cfg = AmericasNLPConfig(
            name=f"{lang}-es",
            version=datasets.Version("1.0.0"),
            description=f"AmericasNLP: {lang}-es",
        )
        configs.append(cfg)
        cfg = AmericasNLPConfig(
            name=f"es-{lang}",
            version=datasets.Version("1.0.0"),
            description=f"AmericasNLP: es-{lang}",
        )
        configs.append(cfg)
    cfg = AmericasNLPConfig(
        name="all-es",
        version=datasets.Version("1.0.0"),
        description="AmericasNLP: all-es",
    )
    configs.append(cfg)
    return configs


class AmericasNLP(datasets.GeneratorBasedBuilder):
    """AmericasNLP Dataset."""

    BUILDER_CONFIGS = _builder_configs()
    BUILDER_CONFIG_CLASS = AmericasNLPConfig

    def _info(self):
        features = datasets.Features(
            {
                "source_sentence": datasets.Value("string"),
                "target_sentence": datasets.Value("string"),
                "source_lang": datasets.Value("string"),
                "target_lang": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        src_lang, trg_lang = self.config.name.split("-")
        src_spanish = src_lang == "es"
        _URL = "https://raw.githubusercontent.com/AmericasNLP/americasnlp2024/master/ST1_MachineTranslation/data/{lang}-spanish/{split}.{lang_code}"
        if "all" not in self.config.name:
            lang = lang_code2lang[trg_lang] if src_spanish else lang_code2lang[src_lang]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "files": dl_manager.download(
                            [
                                _URL.format(
                                    lang=lang,
                                    lang_code=src_lang,
                                    split="train",
                                ),
                                _URL.format(
                                    lang=lang,
                                    lang_code=trg_lang,
                                    split="train",
                                ),
                            ]
                        )
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "files": dl_manager.download(
                            [
                                _URL.format(
                                    lang=lang,
                                    lang_code=src_lang,
                                    split="dev",
                                ),
                                _URL.format(
                                    lang=lang,
                                    lang_code=trg_lang,
                                    split="dev",
                                ),
                            ]
                        )
                    },
                ),
            ]
        else:
            train_files = []
            dev_files = []
            for lang_ in LANGS:
                lang = lang_code2lang[lang_] if src_spanish else lang_code2lang[lang_]
                train_files.append(
                    (
                        _URL.format(
                            lang=lang,
                            lang_code="es" if src_spanish else lang_,
                            split="train",
                        ),
                        _URL.format(
                            lang=lang,
                            lang_code=lang_ if src_spanish else "es",
                            split="train",
                        ),
                    )
                )
                dev_files.append(
                    (
                        _URL.format(
                            lang=lang,
                            lang_code="es" if src_spanish else lang_,
                            split="dev",
                        ),
                        _URL.format(
                            lang=lang,
                            lang_code=lang_ if src_spanish else "es",
                            split="dev",
                        ),
                    )
                )

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "files": dl_manager.download(train_files),
                        # "langs": LANGS,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "files": dl_manager.download(dev_files),
                        # "langs": LANGS,
                    },
                ),
            ]

    def _generate_examples(self, files: list[str]):
        if isinstance(files[0], str):
            with open(files[0], "r", encoding="utf-8") as src_file:
                with open(files[1], "r", encoding="utf-8") as trg_file:
                    for i, (src_sent, trg_sent) in enumerate(zip(src_file, trg_file)):
                        yield (
                            i,
                            {
                                "source_sentence": src_sent.strip(),
                                "target_sentence": trg_sent.strip(),
                            },
                        )
        else:
            lhs, rhs = self.config.name.split("-")
            spa_src = lhs == "es"
            src_lang_files, trg_lang_files = zip(*files)
            src_lines = []
            for i, src_lang_file in enumerate(src_lang_files):
                with open(src_lang_file, "r") as f:
                    for line in f:
                        if not spa_src:
                            src_lines.append((LANGS[i], line.strip()))
                        else:
                            src_lines.append(("es", line.strip()))
            trg_lines = []
            for i, trg_lang_file in enumerate(trg_lang_files):
                with open(trg_lang_file, "r") as f:
                    for line in f:
                        if spa_src:
                            trg_lines.append((LANGS[i], line.strip()))
                        else:
                            trg_lines.append(("es", line.strip()))
            for i, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                yield (
                    i,
                    {
                        "source_lang": src_line[0],
                        "source_sentence": src_line[1],
                        "target_lang": trg_line[0],
                        "target_sentence": trg_line[1],
                    },
                )
