# "nllb-200-distilled-1.3B" 
for MODEL in "nllb-200-distilled-600M" "nllb-200-3.3B"
do
    for SPLIT in "validation" "test"
    do
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=spa_Latn run.dataset=xnli run.dataset_name=es run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=deu_Latn run.dataset=xnli run.dataset_name=de run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=fra_Latn run.dataset=xnli run.dataset_name=fr run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=ell_Grek run.dataset=xnli run.dataset_name=el run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=bul_Cyrl run.dataset=xnli run.dataset_name=bg run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=rus_Cyrl run.dataset=xnli run.dataset_name=ru run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=tur_Latn run.dataset=xnli run.dataset_name=tr run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=arb_Arab run.dataset=xnli run.dataset_name=ar run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=vie_Latn run.dataset=xnli run.dataset_name=vi run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=tha_Thai run.dataset=xnli run.dataset_name=th run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=zho_Hans run.dataset=xnli run.dataset_name=zh run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=hin_Deva run.dataset=xnli run.dataset_name=hi run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=swh_Latn run.dataset=xnli run.dataset_name=sw run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=urd_Arab run.dataset=xnli run.dataset_name=ur run.split=$SPLIT
    done
done
