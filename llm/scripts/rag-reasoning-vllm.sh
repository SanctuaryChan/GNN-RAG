SPLIT=test
DATASET_LIST=RoG-webqsp
MODEL_NAME=llama2
ENGINE=vllm
PROMPT_PATH=prompts/llama2_predict.txt
DATA_PATH=/data/GNN-RAG/datasets
BEAM_LIST=3 # 1 2 3 4 5
MODEL_PATH=/data/GNN-RAG/models/Llama-2-7b-chat-hf
BATCH_SIZE=8
TP_SIZE=2

# RULE_PATH 为LLM本身生成的路径
# RULE_PATH_G1 为GNN生成的路径
# RULE_PATH_G2 为备用（多GNN接口）

#GNN-RAG
for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        RULE_PATH_G1=results/gnn/${DATA_NAME}/rearev-sbert/test.info
        RULE_PATH_G2=None #results/gnn/${DATA_NAME}/rearev-lmsr/test.info

        # no rog
        python src/qa_prediction/predict_answer.py \
            --data_path ${DATA_PATH} \
            --model_name ${MODEL_NAME} \
            --engine ${ENGINE} \
            --model_path ${MODEL_PATH} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --rule_path ${RULE_PATH} \
            --rule_path_g1 ${RULE_PATH_G1} \
            --rule_path_g2 ${RULE_PATH_G2} \
            --predict_path results/KGQA-GNN-RAG/rearev-sbert \
            --batch_size ${BATCH_SIZE} \
            --tensor_parallel_size ${TP_SIZE} \
            --dtype fp16 \
            --gpu_memory_utilization 0.90 \
            --vllm_quiet \
            --vllm_disable_tqdm
    done
done
