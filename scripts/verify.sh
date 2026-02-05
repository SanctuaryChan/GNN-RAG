  python verify_subgraph_alignment.py \
    --gnn-json /data/GNN-RAG/datasets/webqsp/test.json \
    --rog-hf /data/GNN-RAG/datasets/RoG-webqsp \
    --rog-split test \
    --gnn-entities /data/GNN-RAG/datasets/webqsp/entities.txt \
    --gnn-relations /data/GNN-RAG/datasets/webqsp/relations.txt \
    --entity-name-map ../llm/entities_names.json \
    --diff-stats --topk 20 --diff-entity