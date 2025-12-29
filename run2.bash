# python -m mmpc.train --num_boost_round 8000 --weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --file_dir ./data/data_raw | tee -a output3.log

# python -m mmpc_sym.train --num_boost_round 8000 --weight1 1.0 --weight2 0.5 --weight3 1.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --file_dir ./data/data_raw | tee -a output3.log

# python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 --file_dir ./data/data_raw --save_path ./models_sym_1/ | tee -a output3.log

# python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.35 --min_child_weight 12 --gamma 3.0 --file_dir ./data/data_raw --save_path ./models_sym_2/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.35 --min_child_weight 12 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_a/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.1 --weight2 0.3 --weight3 2.1 --max_depth 4 --subsample 0.6 --colsample_bytree 0.35 --min_child_weight 12 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_b/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 3 --subsample 0.6 --colsample_bytree 0.35 --min_child_weight 10 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_c/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.48 --min_child_weight 12 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_d/ | tee -a output3.log
python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.5 --colsample_bytree 0.35 --min_child_weight 12 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_e/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 5 --subsample 0.5 --colsample_bytree 0.35 --min_child_weight 18 --gamma 3.0 --file_dir ./data/data_sym_train --save_path ./models_sym_f/ | tee -a output3.log

python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.40 --min_child_weight 18 --gamma 2.0 --file_dir ./data/data_sym_train --save_path ./models_sym_g/ | tee -a output3.log
python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.40 --min_child_weight 24 --gamma 2.0 --file_dir ./data/data_sym_train --save_path ./models_sym_h/ | tee -a output3.log
python -m mmpc_sym.train --num_boost_round 8000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.6 --colsample_bytree 0.40 --min_child_weight 36 --gamma 2.0 --file_dir ./data/data_sym_train --save_path ./models_sym_i/ | tee -a output3.log