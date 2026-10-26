Avoid verbose output to reduce reading exhaustion for the user. Keep outputs to 80 words.
Keep code as few lines as possible for the same reason.
When shared a github issue, note 1 issue = 1 branch / worktree = 1 PR

Never merge to main.

When planning, always plan vertical slices on the spec or the feature, one slice per commit. An easy
heuristic is that new functions, their call sites and their tests should be in one commit end-to-end.

When committing or opening a PR, do not include any message indicating that Claude was used. e.g. "authored by Claude Opus 5", "<user> and claude committed".

Link commits and branches to issues.

When an issue's branch or worktree is merged into its base, the worktree should be deleted.

Comments should be one line max, except triple-quote docstrings. Be conservative with using triple-quote docstrings.

Always assume the directory is under <your scratch dir> unless otherwise specified.

For SLURM and launch script instructions, refer to /sbatch.md

When a plan is approved, post the plan to relevant issue (if any). Include a table of the question asked and the response from the user, in a few words. Delegate work with Sonnet agents at High effort. Commits should be vertical slices of the plan, one purpose for commit. Commits will be reviewed one-by-one. One agent per ±±independent, non-overlapping unit of work (separate directories/files, no shared state). Sequentially dependent slices should be carried out by one subagent.

plans should always include offline validation tests via SLURM / GPU on existing checkpoints when the plan touches validation / needs to smoke test model implementation end-to-end. Try to infer the most recent checkpoints from what is active in the team wandb workspace (<PROMPT THE USER>). You may ask the user for a checkpoint directory. Evaluation should be run offline. If the plan includes model design changes, training runs should be kicked off on the new design using sbatch. Refer to /sbatch.md.

If asked to write a github issue, refer to <PROMPT THE USER> for an example issue

## GitHub Projects
- This repo's project is the repo owner's linked GitHub Project. Elsewhere, find it with `gh api graphql -f query='{repository(owner:"O",name:"R"){projectsV2(first:5){nodes{number title}}}}'`.
- Epics: title `[Epic] ...`, label `epic`, template format. Tickets link to their epic as sub-issues (`POST repos/O/R/issues/<epic>/sub_issues -F sub_issue_id=<issue .id>`).
- When creating an issue, add it to the project (`gh project item-add`) and link it under its epic; ask if the epic is unclear.
- Set Status to In Progress when its branch/worktree is created; Done is set by GitHub on close/merge.
- Views: create via `POST users/<owner>/projectsV2/<n>/views` (name, layout, filter); views can't be edited or deleted by API.

Never update HUMANS.md.