# | code |NLLB | language
# | ---- |---- | --------
# | ace | Y    | acehnese
# | ban | Y    | banlinese
# | bbc | N    | toba_batak
# | bjn | Y    | banjarese
# | bug | Y    | buginese
# | eng | Y    | english
# | ind | Y    | indonesian
# | jav | Y    | javanese
# | mad | N    | madurese
# | min | Y    | minangkabau
# | nij | N    | ngaju
# | sun | Y    | sundanese


for MODEL in "nllb-200-distilled-600M" "nllb-200-3.3B"
do
    for SPLIT in "validation" "test"
    do
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=ace_Latn run.dataset_name=ace run.split=$SPLIT
        env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=ban_Latn run.dataset_name=ban run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=bjn_Latn run.dataset_name=bjn run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=bug_Latn run.dataset_name=bug run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=ind_Latn run.dataset_name=ind run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=jav_Latn run.dataset_name=jav run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=min_Latn run.dataset_name=min run.split=$SPLIT
        # env HYDRA_FULL_ERROR=1 python -m trident.run experiment=translation run.dataset=indonlp/NusaX-senti run.dir=NusaX-senti run.model=$MODEL "run.text=['text']" run.flores_lang=sun_Latn run.dataset_name=sun run.split=$SPLIT
    done
done
