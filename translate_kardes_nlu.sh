# "nllb-200-distilled-1.3B" 
for MODEL in "nllb-200-distilled-600M" "nllb-200-3.3B"
do
    for SPLIT in "validation" "test"
    do
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=azj_Latn run.dataset="./datasets/kardes_nlu.py" run.dataset_name=xnli.az run.split=$SPLIT "run.text=['premise','hypothesis']" "run.others=['label','premise_english','hypothesis_english']"
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=kaz_Cyrl run.dataset="./datasets/kardes_nlu.py" run.dataset_name=xnli.kk run.split=$SPLIT "run.text=['premise','hypothesis']" "run.others=['label','premise_english','hypothesis_english']"
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=kir_Cyrl run.dataset="./datasets/kardes_nlu.py" run.dataset_name=xnli.ky run.split=$SPLIT "run.text=['premise','hypothesis']" "run.others=['label','premise_english','hypothesis_english']"
    env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=uzn_Latn run.dataset="./datasets/kardes_nlu.py" run.dataset_name=xnli.uz run.split=$SPLIT "run.text=['premise','hypothesis']" "run.others=['label','premise_english','hypothesis_english']"
    done
done
