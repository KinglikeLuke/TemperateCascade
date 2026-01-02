import os


file = "start_ensemble/latin_sh_file_save.txt"


num_lines = sum(1 for line in open(file))


with open(file) as fp:
    for cnt in range(0, num_lines):
        line = fp.readline()
        print(line)

        #iniate job script
        with open("job_submit.sh", "w+") as fh:
            fh.writelines("#!/bin/bash\n\n")

            #specifications of the job that should be submitted
            fh.writelines("#SBATCH --qos=short\n")
            fh.writelines("#SBATCH --job-name=<job_name>\n")
            fh.writelines("#SBATCH --account=<accountname>\n\n")

            fh.writelines("#SBATCH --workdir=out_err_files\n")
            fh.writelines("#SBATCH --output=outfile-%j.txt\n")
            fh.writelines("#SBATCH --error=error-%j.txt\n")
            fh.writelines("#SBATCH --nodes=1\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --time=0-23:50:00\n\n")

            fh.writelines("module load anaconda/5.0.0_py3\n")
            fh.writelines("source activate <conda_environment_name>\n\n")

            #job to be submitted
            fh.writelines("{}".format(line))
            fh.close()

        os.system("sbatch {}".format("job_submit.sh"))

