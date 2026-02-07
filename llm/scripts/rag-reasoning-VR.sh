SPLIT="test"
DATASET_LIST="RoG-cwq"
MODEL_NAME=llama2
PROMPT_PATH=prompts/llama2_predict.txt
DATA_PATH=/data/GNN-RAG/datasets
BEAM_LIST="3" # "1 2 3 4 5"
MODEL_PATH=/data/GNN-RAG/models/Llama-2-7b-chat-hf

# Verbalizer + Reranker configs
RERANKER_PATH=/data/GNN-RAG/saved_models/path_reranker_webqsp
RERANKER_TOPK=5
RERANKER_MAX_LENGTH=256
ENTITIES_NAMES_PATH=/data/GNN-RAG/entities_names.json

  # RULE_PATH 为LLM本身生成的路径
  # RULE_PATH_G1 为GNN生成的路径
  # RULE_PATH_G2 为备用（多GNN接口）

  #GNN-RAG (Verbalizer + Reranker)
  for DATA_NAME in $DATASET_LIST; do
      for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
        RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

        # no rog
        python src/qa_prediction/predict_answer.py \
            --data_path ${DATA_PATH} \
            --model_name ${MODEL_NAME} \
            --model_path ${MODEL_PATH} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --rule_path ${RULE_PATH} \
            --rule_path_g1 ${RULE_PATH_G1} \
            --rule_path_g2 ${RULE_PATH_G2} \
            --predict_path results/KGQA-GNN-RAG/rearev-sbert \
            --batch_size 4 \
            --max_new_tokens 256 \
            --reorder_by_cand \
            --do_sample \
            --use_verbalizer \
            --verbalizer_mode plain \
            --verbalizer_operator auto \
            --reranker_path ${RERANKER_PATH} \
            --reranker_mode cross \
            --reranker_topk ${RERANKER_TOPK} \
            --reranker_max_length ${RERANKER_MAX_LENGTH} \
            --reranker_use_answer \
            --reranker_use_question \
            --entities_names_path ${ENTITIES_NAMES_PATH}
    done
done

# --model_path /data/GNN-RAG/models/rmanluo-RoG \

#GNN-RAG-RA
# for DATA_NAME in $DATASET_LIST; do
#     for N_BEAM in $BEAM_LIST; do
#         RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
#         RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
#         RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

#         python src/qa_prediction/predict_answer.py \
#             --data_path ${DATA_PATH} \
#             --model_name ${MODEL_NAME} \
#             -d ${DATA_NAME} \
#             --prompt_path ${PROMPT_PATH} \
#             --add_rule \
#             --rule_path ${RULE_PATH} \
#             --rule_path_g1 ${RULE_PATH_G1} \
#             --rule_path_g2 ${RULE_PATH_G2} \
#             --model_path rmanluo/RoG \
#             --predict_path results/KGQA-GNN-RAG-RA/rearev-sbert
#     done
# done
