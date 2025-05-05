# 1. Run
To implement the code, please download all the files and ensure you have installed all the packages needed. Then
1. Run file `1_graph_construction.py` to construct the graph from the sample dataset first.
2. Run file `2_No_sequence_detection_EndToEnd_k_fold.py` to see the result.
3. Run file `3_No_sequence_process_perform_file.py` to process the archived data. Please modify the root result path that stores your result from previous step. This will help summarize the performance data to a structured csv file.
# 2. tools
1. `No_sequence_models_EndToEnd_K_fold.py` : models needed for detection task.
2. `tool_EndToEnd_KFolds.py`: includes the functions for evaluation.
3. `early_stop_v1.py`: helps implement the early stop strategy.
