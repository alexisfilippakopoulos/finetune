
OUTPUT_DIR=./results
MODEL=saved_models/summary/xsum_


DATA_PREFIX=./data/summary

for model in 2000; do

python evaluate_cross.py \
  --infile_list ${DATA_PREFIX}/xsum_xlsum_pdfs_cnn \
  --model_name_or_path ${MODEL}${model} \
  --batch_size 8 \
  --world_size 1 \
  --outfile ${OUTPUT_DIR}/ \
  --gpus_per_model 1 \
  --max_new_tokens 60 

done
