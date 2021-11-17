#!/bin/bash

source $HOME/.bashrc
conda activate nllb-llm2vec

# 'flores_passage': "Vérifiez que votre main soit détendue le plus possible lorsque vous jouez chaque note  correctement. Essayez également de ne pas faire de mouvements inutiles avec vos doigts. Vous vous fatigueriez moins de cette façon. N'oubliez pas qu'il n'est pas nécessaire de trop insister sur les touches pour qu'elles résonnent plus fort. C'est le cas sur le piano. Avec l'accordéon, pour produire plus de volume, actionnez le soufflet en augmentant la pression ou la vitesse des mouvements.",
# 'question': 'D’après l’extrait, qu’est-ce qui ne pourrait pas être considéré comme un conseil pertinent pour bien jouer de l’accordéon\xa0?',
# 'mc_answer1': 'Pour produire plus de volume, augmentez la pression exercée sur les touches',
# 'mc_answer2': 'Réduisez le plus possible les mouvements inutiles pour économiser votre énergie',
# 'mc_answer3': 'Faites attention à jouer chaque note en gardant la main détendue',
# 'mc_answer4': 'Actionnez le soufflet plus rapidement pour produire plus de volume',
# 'correct_answer_num': '1',
# 'dialect': 'fra_Latn',$LANG

# LOG_FILE="log_translate_belebele.txt"
# # for MODEL in "nllb-200-distilled-600M" "nllb-200-3.3B"
# for LANG in "acm_Arab" "afr_Latn" "als_Latn" "amh_Ethi" "apc_Arab" "arb_Arab" "arb_Latn" "ars_Arab" "ary_Arab" "arz_Arab" "asm_Beng" "azj_Latn" "bam_Latn" "ben_Beng" "ben_Latn" "bod_Tibt" "bul_Cyrl" "cat_Latn" "ceb_Latn" "ces_Latn" "ckb_Arab" "dan_Latn" "deu_Latn" "ell_Grek" "eng_Latn" "est_Latn" "eus_Latn" "fin_Latn" "fra_Latn" "fuv_Latn" "gaz_Latn" "grn_Latn" "guj_Gujr" "hat_Latn" "hau_Latn" "heb_Hebr" "hin_Deva" "hin_Latn" "hrv_Latn" "hun_Latn" "hye_Armn" "ibo_Latn" "ilo_Latn" "ind_Latn" "isl_Latn" "ita_Latn" "jav_Latn" "jpn_Jpan" "kac_Latn" "kan_Knda" "kat_Geor" "kaz_Cyrl" "kea_Latn" "khk_Cyrl" "khm_Khmr" "kin_Latn" "kir_Cyrl" "kor_Hang" "lao_Laoo" "lin_Latn" "lit_Latn" "lug_Latn" "luo_Latn" "lvs_Latn" "mal_Mlym" "mar_Deva" "mkd_Cyrl" "mlt_Latn" "mri_Latn" "mya_Mymr" "nld_Latn" "nob_Latn" "npi_Deva" "npi_Latn" "nso_Latn" "nya_Latn" "ory_Orya" "pan_Guru" "pbt_Arab" "pes_Arab" "plt_Latn" "pol_Latn" "por_Latn" "ron_Latn" "rus_Cyrl" "shn_Mymr" "sin_Latn" "sin_Sinh" "slk_Latn" "slv_Latn" "sna_Latn" "snd_Arab" "som_Latn" "sot_Latn" "spa_Latn" "srp_Cyrl" "ssw_Latn" "sun_Latn" "swe_Latn" "swh_Latn" "tam_Taml" "tel_Telu" "tgk_Cyrl" "tgl_Latn" "tha_Thai" "tir_Ethi" "tsn_Latn" "tso_Latn" "tur_Latn" "ukr_Cyrl" "urd_Arab" "urd_Latn" "uzn_Latn" "vie_Latn" "war_Latn" "wol_Latn" "xho_Latn" "yor_Latn" "zho_Hans" "zho_Hant" "zsm_Latn" "zul_Latn" 
# do
#     env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset="./datasets/belebele.py" run.dir=belebele run.model=nllb-200-distilled-600M "run.text=['flores_passage','question','mc_answer1','mc_answer2','mc_answer3', 'mc_answer4']" "run.others=['correct_answer_num','link','dialect','ds','question_number']"  run.flores_lang=$LANG run.dataset_name=$LANG run.split=test run.max_length=1024 run.batch_size=32
#     # Check if the command was successful
#     if [ $? -ne 0 ]; then
#         # Log the model and language pair if the command failed
#         echo "nllb-200-distilled-600M $LANG" >> $LOG_FILE
#     fi
#     env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset="./datasets/belebele.py" run.dir=belebele run.model=nllb-200-3.3B "run.text=['flores_passage','question','mc_answer1','mc_answer2','mc_answer3', 'mc_answer4']" "run.others=['correct_answer_num','link','dialect','ds','question_number']"  run.flores_lang=$LANG run.dataset_name=$LANG run.split=test run.max_length=1024 run.batch_size=8
#     # Check if the command was successful
#     if [ $? -ne 0 ]; then
#         # Log the model and language pair if the command failed
#         echo "nllb-200-3.3B $LANG" >> $LOG_FILE
#     fi
# done
for LANG in "ary_Arab"
do
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset="./datasets/belebele.py" run.dir=belebele run.model=nllb-200-distilled-600M "run.text=['flores_passage','question','mc_answer1','mc_answer2','mc_answer3', 'mc_answer4']" "run.others=['correct_answer_num','link','dialect','ds','question_number']"  run.flores_lang=$LANG run.dataset_name=$LANG run.split=test run.max_length=1024 run.batch_size=32
done
