# Advice
*...for those who are about to do a project*

- We are still learning how to design and optimize deep nets.
- Much progress is being made, but it's nevertheless a dark forest.
- Explore if you like, but balance exploration with exploitation of a known good setting, or the closest setting you can find.

## Expertise and Experience

**expertise** *(noun)*
1. skill or knowledge
2. knowing most of the mistakes that can be made in a particular field

**Q: How can you gain expertise?**
- A: Practice on your own with the studios and assignments.
- A: Take our advice that we share today!

## Optimization

**Figure out optimization on one / few / many points, in that order:**
1. Optimize on a single input.
2. Optimize on a batch.
3. Optimize on the training dataset, *or*
   - 3a. optimize on a miniature development dataset, then
   - 3b. optimize on the training set.

The point is to discover issues as soon as possible, or "fail fast," to find out what needs fixing.

**The learning rate and the batch size are more-or-less the cardinal hyperparameters.**
- **Learning rate:** choose the learning rate on a small set if you like (see Bottou).
- **Learning rate:** simplify your life and use a constant rate (no schedule) until everything else is figured out.
- **Batch size:** choose the biggest batch size that can run...
- **Batch size:** ...then try smaller to trade off the number of updates against the size of each update.

## Data

**Know your data!** If you don't, how will you know if your model does?

**Inspect the distribution of the inputs and targets**
- Inspect a random selection of inputs and targets to get a general sense.
- Histogram input dimensions to see range and variability.
- Histogram targets to see range and imbalance.
- Select, sort, and inspect by type of target or whatever else.

**Inspect the inliers, outliers, and neighbors**
- Visualize the distribution and outliers, especially outliers, to uncover dataset issues.
- Look at nearest neighbors in the raw data, in a pre-trained net for the same domain, or in a randomly initialized net.
- Examples: rare grayscale images in a color dataset, huge images that should have been rescaled, corrupted class labels that had been cast to `uint8`.

**Pre-processing: the data as it is loaded is not always the data as it is stored!**
- Inspect the data as it is given to the model by `output = model(data)`.
- *(Slide example: the same coffee-cup image as the original, as loaded by DeCAF (flipped upside down), and as loaded by Caffe (colour channels swapped, so it looks blue).)*

**Pre-processing checks**
- **Summary statistics:** check the min/max and mean/variance to catch mistakes like forgetting to standardize.
- **Shape:** are you certain of each dimension and its size? Sanity-check with dummy data of prime dimensions. Primes share no common factors, so a mistaken reshape, flatten or permute will be more obvious. Example: a 64×64×64×64 array can be permuted without you knowing.
- **Type:** check for casting, especially to lower precision. What's −1 for a byte? How does standardization change integer data?

**Shuffling and dev sets**
- Resample the data to decorrelate; that is, remember to **shuffle**.
- Consider selecting miniature train and test sets for development and testing. Choose these once and keep them fixed throughout optimization and evaluation.
  - It's heartbreaking to wait an entire epoch and only then have your experiment-to-be crash.

## Model

**Keep it as simple as you can!**
- Sure, there are sophisticated models out there...
- ...but they were made by either
  1. going step-by-step, from simple to complex, or
  2. suffering, madness, and the gnashing of teeth.

**Keep it as simple as you can!**
- Do your first experiment with the simplest possible model, with and without your idea.
- At least you will be armed with a little understanding.
- Then follow up with a bidirectional search between simple models and state-of-the-art models, with your own model in the middle.

**Resist snowflake syndrome! Try defaults first.**
> *"But my data/task/idea is special."*

...no it's not, or not until proven so (say, by defeating a residual net).
- Once, while consulting, the lecturer replaced 1+ month of model development with LeNet, and it was more accurate and over 10,000× faster.
- Even if an off-the-shelf net does not prevail, it can serve as a baseline.

**Double-check that the model actually is what you thought you defined.**
```python
# walk the model for inspection
for name_, module_ in model.named_modules():
    if name_ == 'name' or isinstance(module_, nn.Conv2d):
        [...]
```

## Optimization (continued)

**Know your loss! Check the loss against a suitable reference value.**
- Classification with cross-entropy loss: compare against the uniform distribution, e.g. ln(2) ≈ 0.69 for chance on binary classification. *(The slide writes this as −0.69 = ln(0.5).)*
- Regression with squared loss: compare against predicting the mean of the targets (or even just zero).

**Live on the edge and try extreme settings (but just a little bit):**
> If optimization never diverges, your learning rate is too low.

In the style of an Umeshism:
> "If you've never missed a flight, you're spending too much time in airports."

---
*We hope this advice helps you with your next steps and projects!*
