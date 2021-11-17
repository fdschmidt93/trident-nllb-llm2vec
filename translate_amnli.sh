for MODEL in "nllb-200-distilled-600M" "nllb-200-3.3B"
do
    for SPLIT in "validation" "test"
    do
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=quy_Latn run.dataset_name=quy run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=ayr_Latn run.dataset_name=aym run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.model=$MODEL run.flores_lang=grn_Latn run.dataset_name=gn run.split=$SPLIT
    done
done
