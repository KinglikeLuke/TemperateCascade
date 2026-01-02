How to run the skripts:

(1) Run temp_input/timeseries_final.py
(2) Run 00_submit_earth_system_no_enso.py 
(Note: This is a script that usually runs on a high performance computer and submits a very large ensemble to this HPC; individual ensemble members can also be started using the file MAIN_cluster_earth_system_complete_no_enso.py)
(3) For the full ensemble, repeat the results for three different durations. For that change the variable "duration" to 100, 1000, or 50000. Individual saving in results_100, results_1000 and results_50000 is necessary for the evaluation afterwards
(4) Move to the directory evaluations and have a look at the readme.txt (Do the same steps for the results stored results_100 and results_1000)

Note: Create directories in which the results are stored in the first place!


--> If there is any help required, please contact the corresponding author of the study (nico_dot_wunderling_at_pik-potsdam.de)