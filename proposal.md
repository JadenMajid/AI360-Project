Streaming Acoustic Anomaly Detection for Robotic Welding
Oct 19, 2026 · @Chris
Research question
Can pretrained audio embeddings scored by distance to normal welds detect welding defects better than a reconstruction autoencoder, using audio alone? And can the same system flag a defect while the weld is still in progress?
Welding defects are expensive to fix after the fact, so detection during the weld is the practical goal. We test whether the recipe that wins industrial anomaly detection challenges transfers to welding, a loud, broadband sound source unlike typical machine noise.
Related work and novelty
Our novelty is bringing the current machine-sound anomaly detection recipe to welding, and evaluating it as a streaming detector rather than per clip.
Intel welding benchmark (Stemmer et al., 2024). Trains only on good welds. Its audio model is a 1D CNN autoencoder on 192 kHz spectrograms, scored by reconstruction error. Test AUC is 0.846 for audio, 0.898 for video, and 0.918 fused. Audio is weakest on burnthrough (0.70), warping (0.76), excessive penetration and spatter (both 0.78).
DCASE Task 2 (2026 task). Annual challenge on detecting faulty machines from sound with only normal training data. Recent top systems use pretrained audio encoders (BEATs, EAT, CLAP), fine-tune them on auxiliary labels, and score by k-NN distance to normal embeddings. These consistently beat autoencoder baselines, but have not been tested on welding. The 2026 winner, MERL, used BEATs and EAT with a frequency-wise memory bank, deviation-based pooling and score rescaling, scoring 66.2% on the development set against 57.7% for the autoencoder baseline (MERL technical report).
Gap. Intel scores a whole clip by averaging frame scores, so it never measures how early a defect is flagged. Welding sound is loud and broadband, unlike DCASE machines, so it is unclear whether pretrained embeddings transfer.
Dataset
We use the Intel Robotic Welding Multimodal Dataset: 4,040 robotic arc welds from an automotive production floor, with audio, video, sensor time-series and post-weld images. Access is requested; it is research-only.
Split
Good welds
Defective welds
Train
576
0
Validation
122
1,610
Test
121
1,611

Defects span 11 categories, including porosity, burnthrough, spatter and lack of fusion. We reuse Intel's splits so results are directly comparable. Two caveats: labels are the intended defect, not expert-verified, so some label noise is expected; and the paper used 192 kHz audio while the public card lists 16 kHz, so we will re-run their baseline at whatever rate we receive.
Proposed method
We compare four audio-only detectors on identical splits, then extend the best one to streaming.
Baseline. Reimplement Intel's 1D CNN autoencoder on log-spectrograms, scored by reconstruction error. This checks our pipeline against their 0.846.
Pretrained embeddings + k-NN. Extract frame embeddings from a frozen pretrained audio encoder (BEATs or EAT, both transformers pretrained without labels on AudioSet spectrograms). Score each frame by cosine distance to its nearest good-weld embeddings; average frames for a clip score. Compare against plain log-mel features with the same scorer to isolate the encoder's value.
Light fine-tuning. Fine-tune the encoder on good welds with an auxiliary task, classifying weld type and material (available as metadata), as top DCASE systems do. This adapts general audio features to welding sound.
MERL-style scoring. Adopt the scoring backend of the DCASE 2026 winner. A frequency-wise memory bank runs nearest-neighbour search separately in each frequency band and averages the distances, so an unseen mix of normal sounds is not flagged. Relative deviation pooling weights frames that deviate from the clip average, so brief anomalies are not averaged away. Score rescaling adjusts scores by local density. We then average scores from BEATs and EAT. We skip MERL's noise-aware two-microphone model, since the public release is single-channel.
Streaming extension (stretch). Run the best detector on short causal windows and raise an alarm when the running score crosses a threshold set on validation good welds. Report time-to-detection alongside the false alarm rate.
Evaluation. Clip-level AUC overall and per defect category, following Intel. Success means beating 0.846 audio-only; the stretch goal is approaching the 0.918 audio+video result with audio alone.

