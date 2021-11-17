# The way https://huggingface.co/datasets/facebook/belebele is uploaded makes loading data per language extremely slow
# since, for every language, the entire dataset is downloaded as languages are treated as splits and not separate sub datasets

import datasets
import typing as tp
import json

_DESCRIPTION = ""  # TODO

_HOMEPAGE = "https://huggingface.co/datasets/facebook/belebele"  # TODO

_URL = "https://huggingface.co/datasets/facebook/belebele/raw/main/data/{lang}.jsonl"

LANGS = [ "acm_Arab", "afr_Latn", "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "azj_Latn", "bam_Latn", "ben_Beng", "ben_Latn", "bod_Tibt", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "ckb_Arab", "dan_Latn", "deu_Latn", "ell_Grek", "eng_Latn", "est_Latn", "eus_Latn", "fin_Latn", "fra_Latn", "fuv_Latn", "gaz_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hin_Latn", "hrv_Latn", "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kac_Latn", "kan_Knda", "kat_Geor", "kaz_Cyrl", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kin_Latn", "kir_Cyrl", "kor_Hang", "lao_Laoo", "lin_Latn", "lit_Latn", "lug_Latn", "luo_Latn", "lvs_Latn", "mal_Mlym", "mar_Deva", "mkd_Cyrl", "mlt_Latn", "mri_Latn", "mya_Mymr", "nld_Latn", "nob_Latn", "npi_Deva", "npi_Latn", "nso_Latn", "nya_Latn", "ory_Orya", "pan_Guru", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn", "ron_Latn", "rus_Cyrl", "shn_Mymr", "sin_Latn", "sin_Sinh", "slk_Latn", "slv_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "tsn_Latn", "tso_Latn", "tur_Latn", "ukr_Cyrl", "urd_Arab", "urd_Latn", "uzn_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "yor_Latn", "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn", ]


class BelebeleConfig(datasets.BuilderConfig):
    """BuilderConfig for KardesNLU Dataset."""

    def __init__(self, name, **kwargs):
        super(BelebeleConfig, self).__init__(**kwargs)
        self.name = name


def _builder_configs() -> tp.List[BelebeleConfig]:
    configs = []
    for lang in LANGS:
        cfg = BelebeleConfig(
            name=f"{lang}",
            version=datasets.Version("1.0.0"),
            description=f"Belebele: {lang}",
        )
        configs.append(cfg)
    return configs


class Belebele(datasets.GeneratorBasedBuilder):
    """Belebele Dataset."""

    BUILDER_CONFIGS = _builder_configs()
    BUILDER_CONFIG_CLASS = BelebeleConfig

    def _info(self):
        features = datasets.Features(
            {
                "link": datasets.Value("string"),
                "question_number": datasets.Value("int64"),
                "flores_passage": datasets.Value("string"),
                "question": datasets.Value("string"),
                "mc_answer1": datasets.Value("string"),
                "mc_answer2": datasets.Value("string"),
                "mc_answer3": datasets.Value("string"),
                "mc_answer4": datasets.Value("string"),
                "correct_answer_num": datasets.Value("int64"),
                "dialect": datasets.Value("string"),
                "ds": datasets.Value("string"),  # timedate
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        if not self.config.name == "all":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "file": dl_manager.download(
                            _URL.format(
                                lang=self.config.name,
                            )
                        )
                    },
                ),
            ]

    def _generate_examples(self, file: str):
        with open(file, "r", encoding="utf-8") as f:
            for key, line in enumerate(f):
                cur_line = json.loads(line)
                yield key, cur_line
