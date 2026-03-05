# Known Limitations

1. Prompt coverage is broad but not universal.
   - Free-form prompts can still fail if constraints are ambiguous or contradictory.
2. LLM rewrite is non-deterministic without fixed model/version behavior.
   - Use `rules` mode for strict reproducibility.
3. Domain quality depends on checkpoint quality.
   - Weak or undertrained checkpoints reduce realism.
4. The generator can enforce hard filters post-sampling.
   - This can improve constraints but slightly distort joint distributions.
5. Kaggle evaluation is similarity-based, not privacy-safe synthesis certification.
   - Additional privacy and utility checks are needed for production use.
6. GUI is Windows-first for one-click CSV opening (`os.startfile`).

