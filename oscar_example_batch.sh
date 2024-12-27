#SBATCH -J RBC
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:30:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o RBC_job_%j.o
#SBATCH -e RBC_job_%j.e

echo $LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

source /oscar/data/gk/psaluja/jax_env.venv/bin/activate
python3 -u kernel.py