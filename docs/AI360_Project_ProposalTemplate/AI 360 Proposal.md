# **Streaming Acoustic Anomaly Detection for Welding**

## **Background**

[https\://sonibelinstruments.com/](https://sonibelinstruments.com/) is a company that recently graduated from the Hatch incubator at UBC. They received \$2M in funding, hired a founding team and moved to San Francisco.

The premise of their product is this: let’s say you’re building a ship. Think of thousands of welds that need to be done in the process. Typically, the welds are all done together, then a second pass is needed to check the welds for a variety of anomalies. This second pass, considering the personnel and setup costs, is expensive. The solution is detecting anomalies in real-time with machine learning, as the welds are done, by examining the video of the welding arc and the sound.

According to a colleague who interned with them this summer, it appears that the product practically sells itself, and they won contracts with the Royal Canadian Navy among others before they have even piloted their product.

The questions for us are basically:

- Is there data available  
- Can we train a better classifier than what exists  
- Can we make it work in real-time

The answer to 1\) is **yes,** and we have it, 2\) **likely yes,** there’s a decent baseline with outdated methods, and 3\) **likely yes**, this is an engineering challenge and the objective of this company.

If they’re interested, they could provide more data which can only help us against the baseline, or clearer scope. In return, we get a problem that I think is very reasonably attainable in the scope of this class and could result in a fantastic portfolio project.

## **Proof of Concept and Baseline**

**Intel welding benchmark** ([Stemmer et al., 2024](https://arxiv.org/abs/2409.02290v1)). A dataset of both the audio and video streams of welding jobs, classified by “correct”, or 11 different real anomalies. Its audio model is a 1D CNN autoencoder on 192 kHz spectrograms. Test AUC (how well anomaly scores rank defective welds above good ones) is 0.846 for audio, 0.898 for video, and 0.918 fused. Audio is weakest on burnthrough (0.70), warping (0.76), excessive penetration and spatter (both 0.78).

## **The Gap**

**DCASE Task 2** ([2026 task](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)). Annual challenge on detecting faulty machines from sound with only normal training data. Recent top systems use pretrained audio encoders (BEATs, EAT, CLAP), fine-tune them on auxiliary labels, and score by k-NN distance to normal embeddings. These consistently beat autoencoder baselines, **but have not been tested on welding**. The 2026 winner, MERL, used BEATs and EAT with a frequency-wise memory bank, deviation-based pooling and score rescaling, scoring 66.2% on the development set against 57.7% for the autoencoder baseline ([MERL technical report](https://shadow.merl.com/publications/TR2026-100)).

In short, **take the Intel dataset and beat their baseline with MERL’s method.**

## **Dataset**

We use the [Intel Robotic Welding Multimodal Dataset](https://huggingface.co/datasets/IntelLabs/Intel_Robotic_Welding_Multimodal_Dataset): 4,040 robotic arc welds from an automotive production floor, with audio, video, sensor time-series and post-weld images. Access must be requested, but I already received approval and have the dataset.

| Split | Good welds | Defective welds |
| ----- | ----- | ----- |
| Train | 576 | 0 |
| Validation | 122 | 1,610 |
| Test | 121 | 1,611 |

Defects span 11 categories, including porosity, burnthrough, spatter and lack of fusion. We reuse Intel's splits so results are directly comparable. Two caveats: labels are the intended defect, not expert-verified, so some label noise is expected; and the paper used 192 kHz audio while the public card lists 16 kHz, so we will re-run their baseline at whatever rate we receive.

## **Proposed method**

1. **Baseline.** Reimplement Intel's 1D CNN autoencoder on log-spectrograms, scored by reconstruction error. This checks our pipeline against their 0.846.  
2. **Pretrained embeddings \+ k-NN.** Extract frame embeddings from a frozen pretrained audio encoder (BEATs or EAT, both transformers pretrained without labels on AudioSet spectrograms). Score each frame by cosine distance to its nearest good-weld embeddings; average frames for a clip score. Compare against plain log-mel features with the same scorer to isolate the encoder's value.  
3. **Light fine-tuning.** Fine-tune the encoder on good welds with an auxiliary task, classifying weld type and material (available as metadata), as top DCASE systems do. This adapts general audio features to welding sound.  
4. **MERL-style scoring.** Adopt the scoring backend of the DCASE 2026 winner. A frequency-wise memory bank runs nearest-neighbour search separately in each frequency band and averages the distances, so an unseen mix of normal sounds is not flagged. Relative deviation pooling weights frames that deviate from the clip average, so brief anomalies are not averaged away. Score rescaling adjusts scores by local density. We then average scores from BEATs and EAT. We skip MERL's noise-aware two-microphone model, since the public release is single-channel.  
5. **Extension to real-time (stretch).** Run the best detector on short causal windows and raise an alarm when the running score crosses a threshold set on validation good welds. Report time-to-detection alongside the false alarm rate.   
6. **Extension to video \+ audio (stretch)**. Fuse audio and video anomaly scores, as Intel did, to test whether stronger audio features narrow the gap to video.  
7. **Extension to specific anomaly classification (stretch).** There exists a [preprint](https://www.researchsquare.com/article/rs-10675288/v1?redirect=/article/rs-10675288) on this dataset that fuses post-weld images, video, audio and telemetry for supervised defect-type classification (92% macro-F1). We would train a defect-type classifier on our audio embeddings using part of the validation defects, and compare against it.

**Evaluation.** Clip-level AUC overall and per defect category, following Intel. Success means beating 0.846 audio-only; the stretch goal is approaching the 0.918 audio+video result with audio alone.

