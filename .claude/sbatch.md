This skill is invoked during planning or grilling. When a plan is implemented, set up a new SLURM script to test the changes in an actual training run. Reference other run scripts for config. General blueprint:

#SBATCH --partition=<PROMPT THE USER>
#SBATCH --job-name=<issue-#-desc>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --mem=32gb

If the new code contains an important toggle, ask the user if they want several runs. For each run, arm a monitor and if it crashes, diagnose and relaunch.

Checkpoints should never be saved on worktrees. Save them in the base git repo, and when possible, name them according to their readable name and not their wandb id. (e.g. Correct: `47 - new_rope_uniform`, Incorrect: `2c02s2pu`)

SLURM names should have their issue number and then 5 chars distinguishing from other runs. (e.g. Correct: `78-sm41` = issue 78, small-medium model size, training policy 4.1)

Default partition: <PROMPT THE USER>. Partition for larger models: <PROMPT THE USER>.

Never pass overrides at the sbatch command line (e.g. `CKPT=... sbatch script.sh`); edit the script's export lines instead. scripts/**/*.sh and configs/experiment/* are gitignored (personal, local-path scripts), so this isn't a git paper trail — the record of what a run actually used is the config wandb logs at start ("Logging hyperparameters!"), not git history. Training/eval data (online-generated sessions or static overfit sessions) should be staged in /dev/shm (RAM), not /scratch-ssd or NFS — CPU/RAM are already reserved by the job regardless.

Every sbatch run's name (wandb `logger.wandb.name`) must be prepended with the issue it's working on, in this format: `<issue> - <name>` (e.g. `47 - new_rope_uniform`).

When a plan is approved, post the plan to relevant issue (if any) and delegate work with Sonnet agents at High effort. Commits should be vertical slices of the plan, one purpose for commit. Commits will be reviewed one-by-one. One agent per independent, non-overlapping unit of work (separate directories/files, no shared state). Sequentially dependent slices should be carried out by one subagent.