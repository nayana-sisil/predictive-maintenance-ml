# Predictive Maintenance Using Machine Learning

**Preventing equipment failures before they happen because nobody likes unexpected downtime.**

## What's This About?

Industrial machines break down. It's a fact of life. But what if we could predict these failures before they actually happen? That's exactly what this project does. Using machine learning on operational sensor data, we can catch warning signs early and schedule maintenance at the right time – not too early (wasting money) and definitely not too late (hello, expensive breakdowns).

## Check It Out

**Live Demo**: [Try it on Hugging Face](https://huggingface.co/spaces/nayanasisil2700/predictive-maintenance)

Go ahead, play around with it. It's pretty satisfying to see the model predict failures.

## The Problem

Traditional maintenance strategies are basically guesswork dressed up as process:
- **Time based maintenance**: Change parts every X hours whether they need it or not (wasteful)
- **Run to failure**: Wait until something breaks, then panic (expensive and disruptive)
- **Result**: Either throwing money away on unnecessary maintenance or dealing with catastrophic failures at 2 AM

There had to be a better way.

## The Solution

Train a machine learning model to recognize patterns in sensor data that precede failures. The tricky part? Failures are rare (which is good for operations but challenging for ML). Only 3.4% of our data points represent actual failures.

This imbalance meant standard ML approaches wouldn't work – the model would just predict "everything's fine" 96.6% of the time and call it a day.

## Dataset Details

I worked with the **AI4I 2020 Predictive Maintenance Dataset** from UCI. It's synthetic but realistically modeled on actual industrial data (companies don't like sharing when their equipment fails, for obvious reasons).

**What's inside**:
- 10,000 machine operation snapshots
- 339 failures (3.4% – classic imbalanced dataset)
- Multiple sensor readings: temperatures, rotation speed, torque, tool wear
- Three product quality levels: Low, Medium, High

**The interesting bit**: We only get a binary failure label, not which specific component failed. Just like real life – you know something's about to break, but not exactly what.

## How I Approached This

### Step 1: Understanding the Data

First, I dug into the data to understand what was actually happening. Some interesting findings:

- Failures cluster at extreme values of torque and rotational speed – machines have a "happy zone" and bad things happen outside it
- Low-quality products fail more often (235 failures) compared to high-quality ones (21 failures)
- Air and process temperatures are highly correlated (0.88) – they move together
- Torque has the strongest relationship with failures, though it's still only 0.19 correlation

The correlation heatmap was enlightening but also humbling – no single feature screamed "I predict failures!" This meant I'd need the full power of ensemble methods.

### Step 2: Data Preprocessing

Cleaned up the data and dealt with some quirks:

- Encoded product types (Low → 0, Medium → 1, High → 2)
- Used **RobustScaler** for features with outliers (rotational speed and torque)
- Applied **MinMaxScaler** to the rest
- Kept the train-test split stratified so the failure ratio stayed consistent

### Step 3: Tackling the Imbalance

This is where things got interesting. I tested multiple approaches:

**Balanced Ensemble Models**:
- Balanced Random Forest
- Balanced Bagging
- RUSBoost
- Easy Ensemble

**Traditional Models + Sampling Tricks**:
- Random Forest with various sampling techniques
- Bagging Classifier with different resampling methods

I tried six different sampling approaches – RandomOverSampler, SMOTE, BorderlineSMOTE, ClusterCentroids, TomekLinks, and NearMiss. Each has its own philosophy about handling imbalance.

**The winner?** Bagging with TomekLinks. It removes noisy boundary samples, helping the model learn cleaner decision boundaries.

## Results That Actually Matter

### Model Performance

Here's what the top three models achieved:

| Model | Macro F1 | Precision | Recall | What This Means |
|-------|----------|-----------|--------|-----------------|
| **Bagging + TomekLinks** | **0.88** | **0.85** | **0.71** | Catches 71% of failures with few false alarms |
| Random Forest + RandomOverSampler | 0.86 | 0.85 | 0.65 | Solid but misses more failures |
| Balanced Bagging | 0.69 | 0.28 | 0.88 | Catches almost everything but cries wolf constantly |

The Bagging + TomekLinks model is technically the best – highest F1 score, great precision, respectable recall. It's the model you'd put in a research paper.

But here's where it gets interesting...

### The Money Talk

I ran a cost-benefit analysis because, let's be honest, that's what actually matters in the real world.

**Assumptions** (based on typical Sri Lankan industrial costs):
- Each inspection: Rs. 500
- Each breakdown: Rs. 20,000 (includes downtime, emergency repairs, lost production)

**Baseline scenario** (current run-to-failure approach): Rs. 1,700,000 in total breakdown costs

**With ML Models**:

| Model | Total Cost | Savings | Efficiency |
|-------|-----------|---------|------------|
| Bagging + TomekLinks | Rs. 535,500 | Rs. 1,164,500 | High precision, fewer inspections |
| Random Forest + RandomOverSampler | Rs. 632,500 | Rs. 1,067,500 | Good balance |
| Balanced Bagging | Rs. 334,500 | Rs. 1,365,500 | Maximum savings, many inspections |

**Plot twist**: The "technically inferior" Balanced Bagging model saves the most money. Why? Because preventing a single Rs. 20,000 breakdown pays for 40 unnecessary Rs. 500 inspections.

## Which Model Should You Actually Use?

This depends entirely on your situation:

**Scenario A: You have maintenance capacity**
- Go with Balanced Bagging
- Yes, you'll do more inspections (some unnecessary)
- But you'll prevent almost every failure
- **Savings: Rs. 1.36 Million**

**Scenario B: Your maintenance team is stretched thin**
- Use Bagging with TomekLinks
- Fewer false alarms means less wasted effort
- Still prevents most failures
- **Savings: Rs. 1.16 Million**

**Scenario C: You're obsessed with technical elegance**
- Also Bagging with TomekLinks
- Best F1 score, highest PR-AUC
- Makes your technical documentation look good

## Technical Stack

Built with the usual suspects:
- pandas & numpy for data wrangling
- scikit-learn for modeling
- imbalanced-learn for handling class imbalance
- matplotlib & seaborn for visualization (spent way too much time making those plots look good)
- pickle for saving models (yes, really)

## What I Learned

1. **The best model isn't always the "best" model**: Technical metrics don't always align with business value
2. **Imbalanced data is genuinely hard**: You can't just throw more data at it
3. **TomekLinks is underrated**: Everyone talks about SMOTE, but noise removal worked better here
4. **Context matters**: The optimal model depends on operational constraints, not just F1 scores

## Future Ideas

If I revisit this (or if someone wants to extend it):
- Real-time prediction pipeline with streaming sensor data
- LSTM networks for true time-series modeling (this dataset is really snapshots, not sequences)
- Anomaly detection for catching novel failure modes
- Integration with maintenance management systems
- Maybe even a mobile app for maintenance teams in the field

## Run It Yourself

```bash
git clone https://github.com/nayana-sisil/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
```

Then open the notebook and start exploring. All the models are saved in the `models/` folder if you want to load and test them directly.

## Let's Connect

I'm always up for discussing ML projects, predictive maintenance, or why your model isn't working:

- **GitHub**: [@nayana-sisil](https://github.com/nayana-sisil)
- **LinkedIn**: [Nayana Sisil](https://www.linkedin.com/in/nayanasisil/)
- **Email**: nayanasisil@gmail.com
- **WhatsApp**: +94 76 860 9939

Feel free to reach out if you have questions, suggestions, or just want to chat about machine learning.

## License & Attribution

Dataset from UCI Machine Learning Repository (CC BY 4.0). Thanks to them for making quality datasets publicly available.

---

**Disclaimer**: This is a portfolio project. If you're deploying this in production, please involve domain experts, do proper testing, and maybe reconsider your life choices if critical equipment depends solely on my code.
