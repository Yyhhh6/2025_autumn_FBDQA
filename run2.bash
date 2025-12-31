python -m mmpc.train --num_boost_round 4000 --weight1 1.5 --weight2 0.5 --weight3 1.5 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 18 --gamma 4.3 | tee -a output_ZZZ.log

python -m mmpc.train --num_boost_round 4000 --weight1 1.5 --weight2 0.5 --weight3 1.5 --max_depth 3 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 12 --gamma 3.0 | tee -a output_ZZZ.log

python -m mmpc.train --num_boost_round 4000 --weight1 2.0 --weight2 0.5 --weight3 2.0 --max_depth 4 --subsample 0.5 --colsample_bytree 0.48 --min_child_weight 12 --gamma 4.3 | tee -a output_ZZZ.log