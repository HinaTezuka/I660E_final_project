bash
source ~/.bashrc
cd ~/I660E_final_project
conda activate i660_reimplementation
module load cuda/12.1
python /home/s2410121/I660E_final_project/eval_amazon_beauty.py
conda deactivate
