How to run the skripts:

--> Map Safe Operating Space
(1) data_preparator_safe_op_space_submit.sh runs the file "data_preparator_safe_op_space.py" (runs approximately 12-16 hours)
(2) safe_op_space.py
(3a) safe_op_space_post.py
(3b) safe_op_space_tipping_risk.py
[COMMENT: (3a) and (3b) can be run in parallel]


--> Map Overshoot strength
(1) data_preparator_overshoot_submit.py uses the file "network_starter.txt" --> this submits the script "data_preparator_overshoot.py" (takes 50-60 hours per file)
(2a) overshoot_evaluator.py
(2b) overshoot_hist.py
[COMMENT: (2a) and (2b) can be run in parallel]


Note: Create directories in which the results are stored in the first place!