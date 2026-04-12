# Module 1: Data Pipelines & Visualisation Mastery with Python — Speaker Notes

Total time: ~180 minutes (3 hours)

---

## Slide 1: Title

**Time**: ~1 min
**Talking points**:

- Welcome to Module 1 of the ASCENT — ML Engineering from Foundations to Mastery at Terrene Open Academy.
- Read the provocation aloud: "Can you trust a number you didn't explore yourself?"
- Let it sit for a moment. This is the theme of the entire module.
- If beginners look confused: "Today you will go from zero Python to loading a dataset with 15 million rows. By the end, you will have skills no spreadsheet can match."
- If experts look bored: "Even experienced practitioners will find the Kailash engine integrations and the Polars-native approach worth their attention. The theory sections include information geometry and exponential families."
  **Transition**: "Let me show you exactly what you will be able to do by the end of today."

---

## Slide 2: What You Will Learn Today

**Time**: ~3 min
**Talking points**:

- Walk through the left column — these are concrete skills, not abstract concepts.
- Emphasise the three depth layers. Reassure the room: "FOUNDATIONS is for everyone. If you are new to programming, follow the green markers. If you are already comfortable with Python, the blue THEORY and purple ADVANCED slides will challenge you."
- Point out that nobody is expected to follow everything. The three layers let everyone learn at their own pace.
- If beginners look confused: "The FOUNDATIONS track alone will make you productive with data. You will not need to understand the maths to complete the exercises."
- If experts look bored: "Stick around for the MLE derivations, exponential families, and information geometry slides — they go deeper than most university courses."
  **Transition**: "Here is the roadmap for our 8 lessons today."

---

## Slide 3: Your Journey: 8 Lessons

**Time**: ~2 min
**Talking points**:

- Walk through the table quickly — do not read every cell. Highlight the progression: start with basics, end with a full pipeline.
- Note that Lessons 1.1-1.3 are pure Python and Polars. Lessons 1.4-1.6 are data manipulation at scale. Lessons 1.7-1.8 introduce Kailash engines.
- "By Lesson 1.8, you will run a complete exploratory data analysis on a real dataset — the same workflow a professional data scientist uses."
  **Transition**: "To do all that, you will use three Kailash ML engines."

---

## Slide 4: Kailash Engines You Will Meet

**Time**: ~2 min
**Talking points**:

- Introduce each engine briefly: DataExplorer profiles your data. PreprocessingPipeline cleans it. ModelVisualizer creates charts. ConnectionManager stores results.
- Emphasise: "You will learn the manual way first, then see how the engine automates it. That way, when the engine gives you an unexpected result, you know how to debug it."
- If beginners look confused: "Think of these as power tools. We will teach you the hand-tool version first so you understand what the power tool is doing."
  **Transition**: "Before we dive in, let us make sure everyone's environment is set up."

---

## Slide 5: Tools and Setup

**Time**: ~3 min
**Talking points**:

- Show the three exercise formats. Emphasise: everyone picks ONE format and sticks with it today.
- If anyone has not set up their environment, now is the time. Walk through: `uv sync` for local, or open Colab links.
- Common question: "Which format should I use?" Answer: "Local .py if you want the developer experience. Jupyter if you prefer cells. Colab if you do not want to install anything."
  **Transition**: "Now let me tell you a story about why everything we are about to learn matters."

**[PAUSE FOR QUESTIONS — 2 min]**

---

## Slide 6: The Story That Starts Everything — HDB Flash Crash

**Time**: ~4 min
**Talking points**:

- Tell this as a story, not a lecture. Build tension: prices rising 10 quarters straight, then a sudden anomaly nobody saw.
- Key point: dashboards showed nothing unusual because they used aggregated monthly medians.
- "The anomaly was only caught by someone who did what we are about to learn — loaded the raw data and explored it themselves."
- If beginners look confused: "A dashboard is like a weather report. It tells you the average temperature. But it does not tell you that it hailed in your neighbourhood."
- If experts look bored: "This is a textbook example of aggregation masking distributional anomalies. We will formalise this when we cover bimodal distributions."
  **Transition**: "So what exactly went wrong?"

---

## Slide 7: What Went Wrong? — Dashboard Blindness

**Time**: ~3 min
**Talking points**:

- Walk through the three failure modes: aggregation hides outliers, pre-built charts cannot ask new questions, no distribution view.
- The right column shows what EDA would have caught: bimodal distribution, temporal clustering, correlation with flat type.
- "This is why the module is called Data Pipelines and Visualisation MASTERY. Mastery means you can find things nobody asked you to look for."
  **Transition**: "And this was not an abstract problem."

---

## Slide 8: This Affected YOUR Property Value

**Time**: ~3 min
**Talking points**:

- Make it personal. "If you own an HDB flat in Queenstown, Toa Payoh, or Ang Mo Kio, your valuation was affected."
- Point to the code snippet. "Three lines of Polars. That is all it took to find the anomaly. By the end of today, you will be able to write these three lines yourself."
- If beginners look confused: "Do not worry about the code yet. We will build up to it step by step."
- If experts look bored: "The root cause — subsidised transfers entering the public dataset without classification — is a data governance failure. Module 6 covers the governance side."
  **Transition**: "Let us start with the absolute basics. What IS data?"

---

## Slide 9: What Is Data?

**Time**: ~3 min
**Talking points**:

- Start extremely simple. "Every time someone buys an HDB flat, a record is created. That record is one row in a table."
- Walk through the four types: numeric, categorical, ordinal, temporal. Use the examples in the table.
- Common question: "What about boolean data?" Answer: "It is a special case of categorical with two values. We will see it when we learn about filtering."
- If beginners look confused: "Think of a spreadsheet. Rows are individual records. Columns are properties of those records."
  **Transition**: "Now, how do we work with this data? We use a language called Python."

---

## Slide 10: What Is Python?

**Time**: ~4 min
**Talking points**:

- Emphasise Python is plain English-like syntax. Read the code example aloud.
- The Excel comparison table is powerful for this audience. Let the numbers speak: 1M row limit in Excel vs unlimited in Python. 2 seconds to load the HDB dataset vs "Cannot open."
- If beginners look confused: "You do not need to memorise anything. The exercises guide you step by step. Just follow along."
- If experts look bored: "If you already know Python, use this time to glance ahead at the theory slides."
  **Transition**: "Let us write our first Python."

---

## Slide 11: Variables, Strings, and Numbers

**Time**: ~4 min
**Talking points**:

- Walk through the code slowly. Explain: variables are named containers. Strings are text in quotes. Numbers are integers and decimals.
- f-strings: "The f before the quote means you can embed variables directly. This is the most useful Python feature for data exploration."
- Arithmetic: price per square metre is a real calculation a property agent would do.
- If beginners look confused: "A variable is like a labelled jar. You put something in it, and later you can get it back by name."

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: create variables, compute price per sqm, print results.
  **Transition**: "Now let us load real data."

---

## Slide 12: What Is Polars?

**Time**: ~4 min
**Talking points**:

- "Polars is what we use instead of pandas. It is faster, uses less memory, and the entire ASCENT programme uses it."
- Walk through: `pl.read_csv()`, `.shape`, `.columns`, `.head()`, `.describe()`.
- "describe() is your first diagnostic on any dataset. Run it before anything else."
- Kailash connection callout: DataExplorer does describe() plus 50 more diagnostics. But first you need to know what each one means.
- If beginners look confused: "A DataFrame is just a table. Rows and columns, like a spreadsheet but in code."
  **Transition**: "When you look at a column of numbers, the most important thing to understand is its distribution."

---

## Slide 13: What Is a Distribution?

**Time**: ~3 min
**Talking points**:

- "A distribution answers: if I pick a random HDB transaction, what price am I likely to see?"
- Walk through the three shapes: bell curve, right-skewed, bimodal.
- Connect back to the opening case: "The HDB flash crash created a bimodal distribution. The mean looked normal; the histogram screamed. This is why you ALWAYS plot before you compute."
  **Transition**: "Once you can see the shape, you need to measure its centre."

---

## Slide 14: Mean, Median, Mode

**Time**: ~3 min
**Talking points**:

- Walk through the code example with the outlier ($2.5M penthouse).
- "The mean is $772K — that is misleading. The median is $370K — that is representative."
- "Singapore property reports use the median. Now you know why."
- If beginners look confused: "Mean is the average. Median is the middle value when you sort. For income and property prices, always use median."
  **Transition**: "Do two things move together? That is correlation."

---

## Slide 15: What Is Correlation?

**Time**: ~3 min
**Talking points**:

- Positive: bigger flat = higher price. Negative: older lease = lower price. No correlation: block number tells you nothing.
- "Correlation is NOT causation. Ice cream and drowning are correlated because of hot weather, not because one causes the other."
- Show the scatter plot code. "Always plot first, compute later."
  **Transition**: "Now let us see how Kailash automates what we just learned."

---

## Slide 16: What Is Kailash?

**Time**: ~3 min
**Talking points**:

- "Kailash is the Terrene Foundation's open-source SDK. It is not a tool — it is a platform for building ML systems."
- Walk through the comparison table: 50 lines of glue code vs 3 lines with Kailash.
- "You learn the concepts first, then Kailash automates them. This module teaches both paths."
  **Transition**: "Back to Python fundamentals. How do we filter data?"

---

## Slide 17: Booleans and Filtering

**Time**: ~4 min
**Talking points**:

- Booleans are yes/no answers. Comparison operators are how you ask questions of your data.
- Walk through the Polars filter example: all Queenstown transactions above $500K.
- "The ampersand & means AND — both conditions must be true. The pipe | means OR — either condition."
- If beginners look confused: "Think of filtering as asking a question. 'Show me only the rows where town is Queenstown AND price is above $500K.'"

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: filter by town, filter by price range, combine conditions.
  **Transition**: "Now let us create new columns from existing ones."

---

## Slide 18: Transforming Data with with_columns()

**Time**: ~3 min
**Talking points**:

- `with_columns()` adds new columns. `.alias()` names them.
- Method chaining reads like a sentence: load, filter, transform, sort, show top 10.
- "This is the most important pattern in Polars. Get comfortable with it — you will use it hundreds of times."
  **Transition**: "If you find yourself writing the same code twice, it is time to learn functions."

---

## Slide 19: Functions: Reusable Code

**Time**: ~4 min
**Talking points**:

- "A function is a recipe. You write it once, then call it with different ingredients."
- Walk through the anatomy: def, parameters, docstring, body, return.
- Emphasise indentation: "Python uses 4 spaces to define blocks. If your code is not indented correctly, it will not work."
- If beginners look confused: "You do not need to write complex functions today. Just understand the pattern: name, inputs, body, output."
  **Transition**: "The most powerful pattern in data analysis is group-by plus aggregate."

---

## Slide 20: Aggregation: group_by() + agg()

**Time**: ~4 min
**Talking points**:

- "For each town, what is the median resale price? That is a group_by question."
- Walk through the code: group_by("town"), then multiple aggregations in agg().
- Show the table of available aggregations. "mean, median, std, min, max, count, sum, quantile — these are your basic toolkit."
- If beginners look confused: "Group-by is like sorting papers into piles by category, then counting each pile."

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: group by town, compute stats, sort by median price.
  **Transition**: "What if your data lives in multiple tables?"

---

## Slide 21: Joins: Combining Datasets

**Time**: ~4 min
**Talking points**:

- "HDB prices in one table, MRT distances in another. To ask 'Do flats near MRT cost more?', you must join."
- Left join keeps all HDB rows. Inner join keeps only matching rows.
- "Think of a join like VLOOKUP in Excel, but it works on millions of rows in under a second."
- If beginners look confused: "Imagine two address books. A join finds everyone who appears in both."
  **Transition**: "Now let us compute trends over time."

---

## Slide 22: Window Functions: Trends Over Time

**Time**: ~4 min
**Talking points**:

- "A window function computes a value for each row based on a group of related rows — without collapsing the data."
- `.over("town")` means "compute this separately for each town." That is the window.
- Rolling average, year-on-year change, rank within group — three essential window operations.
- If beginners look confused: "Group-by collapses your data into one row per group. Window functions keep all rows but add a computed column."

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "We have covered the foundations. Now let us see the mathematics behind these ideas."

---

## Slide 23: From Histograms to Probability (THEORY)

**Time**: ~3 min
**Talking points**:

- "The histogram is a rough draft. As you increase bins and data, it converges to a smooth curve — the probability density function."
- The integral equation: the area under the curve between two points is the probability of falling in that range.
- "The taller the curve at a point, the more likely values are near there. The total area is always 1."
- If beginners look confused: "Just remember: taller bar = more data points in that range. That is the core idea."
- If experts look bored: "We are building toward the exponential family formulation in a few slides."
  **Transition**: "The most important distribution in statistics is the Normal distribution."

---

## Slide 24: The Normal Distribution (THEORY)

**Time**: ~3 min
**Talking points**:

- Show the formula. "Two parameters: mu shifts left/right, sigma stretches/compresses."
- 68-95-99.7 rule: "If HDB prices were Normal with mu=$500K and sigma=$80K, 95% would fall between $340K and $660K."
- "In reality, HDB prices are right-skewed, not Normal. But the Normal is still the building block for most statistical methods."
  **Transition**: "Not everything is bell-shaped. Let us see two more distributions."

---

## Slide 25: Poisson and Exponential (THEORY)

**Time**: ~3 min
**Talking points**:

- Poisson: "How many HDB transactions per day?" Counts of events in fixed intervals.
- Exponential: "How long until the next transaction?" Time between events.
- "These two are two sides of the same coin. If arrivals are Poisson, waiting times are Exponential."
- Mean = Variance = lambda for Poisson. Memoryless property for Exponential.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "All these distributions are actually members of one family."

---

## Slide 26: The Exponential Family (THEORY)

**Time**: ~4 min
**Talking points**:

- "One formula covers Normal, Poisson, Exponential, Bernoulli, Gamma, Beta."
- Key insight: T(x) is the sufficient statistic — it captures ALL information about theta.
- MLE has a closed-form solution for exponential family distributions.
- If beginners look confused: "This is optional theory. The key takeaway is that these distributions are related, and that is why the same techniques work across all of them."
- If experts look bored: "The log-partition function generates all moments. This connects directly to GLMs and natural exponential family theory."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Let us go deeper into sufficient statistics."

---

## Slide 27: Sufficient Statistics (THEORY)

**Time**: ~3 min
**Talking points**:

- "A sufficient statistic captures all information about the parameter. Once you know T(x), the raw data adds nothing."
- For Normal with known variance: the sample mean is sufficient for mu.
- "This is why describe() is so powerful — the summary statistics often ARE the sufficient statistics."
- Fisher-Neyman factorisation theorem gives the formal test.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Given data, how do we find the best parameters? Maximum Likelihood Estimation."

---

## Slide 28: Maximum Likelihood Estimation (THEORY)

**Time**: ~4 min
**Talking points**:

- "MLE asks: which parameter values make this exact data most likely to have occurred?"
- Walk through the likelihood function and the log transformation.
- "We take the log because products of tiny numbers underflow. The log turns products into sums."
- If beginners look confused: "The plain English version: MLE finds the parameter that best fits your data."
  **Transition**: "Let us derive the MLE for the Normal distribution step by step."

---

## Slide 29: MLE for the Normal — Step by Step (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through Steps 1-2 slowly. The log-likelihood, differentiation, setting to zero.
- "The MLE for the mean is the sample mean. Intuitive and exact."
- If beginners look confused: "Skip the derivation details. The result is what matters: the best estimate of the average is the sample average."
  **Transition**: "Step 3 gives us the variance."

---

## Slide 30: MLE for the Normal (continued) (THEORY)

**Time**: ~3 min
**Talking points**:

- "The MLE for variance divides by n, not n-1. It is biased — slightly underestimates true variance."
- Production gotcha: Polars `.var()` uses n-1 by default (Bessel's correction). Show how to compute MLE variance.
- "For large n, the difference is negligible. For small samples, it matters."
  **Transition**: "How good can an estimator be? Fisher Information sets a floor."

---

## Slide 31: Fisher Information (THEORY)

**Time**: ~3 min
**Talking points**:

- "Fisher Information measures how sharp the likelihood peak is. Sharp = data strongly points to one parameter."
- For Normal: I(mu) = n/sigma^2. More data or less noise = more information.
- Cramer-Rao Bound: no unbiased estimator can have lower variance than 1/I(theta).
- If beginners look confused: "Think of it this way: more data = more certainty = better estimates. Fisher Information quantifies exactly how much."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Here are the properties that make MLE the default choice."

---

## Slide 32: MLE Properties (THEORY)

**Time**: ~2 min
**Talking points**:

- Consistent, asymptotically normal, asymptotically efficient, invariant.
- "MLE is the workhorse of statistics because it has all these desirable properties as your sample grows."
- When MLE fails: small samples, misspecified model, multimodal likelihood.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "For the experts, let us visit information geometry."

---

## Slide 33: Information Geometry (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "Distributions live on a curved surface. The Fisher Information Matrix defines the natural distance on this surface."
- KL divergence is the local quadratic approximation of this Riemannian distance.
- "Natural gradient descent, used in modern RL and deep learning, follows the Fisher metric instead of the Euclidean gradient."
- Connection to Module 6: PPO and TRPO use KL divergence constraints.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Now let us add prior knowledge to our estimation."

---

## Slide 34: Bayesian Thinking: Prior to Posterior (THEORY)

**Time**: ~4 min
**Talking points**:

- "You have a belief about HDB prices before looking at data. You observe data. Your updated belief is the posterior."
- Walk through Bayes' theorem: prior, likelihood, posterior.
- "The posterior is always a compromise between prior and data. With lots of data, the posterior is dominated by the likelihood."
- If beginners look confused: "Think of it like updating your weather forecast when you see dark clouds. Your initial guess changes based on new evidence."
  **Transition**: "When the maths works out perfectly, we get conjugate priors."

---

## Slide 35: Conjugate Priors (THEORY)

**Time**: ~3 min
**Talking points**:

- Normal-Normal: posterior mean is a precision-weighted average. "The formula looks complex, but the idea is simple — the posterior is a weighted average of what you believed and what the data shows."
- Beta-Binomial: "Just add successes to alpha and failures to beta. Beautifully simple."
- Why "conjugate"? Prior and posterior have the same family. Data only updates parameters.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "MAP estimation bridges MLE and Bayesian thinking."

---

## Slide 36: MAP Estimation (THEORY)

**Time**: ~3 min
**Talking points**:

- "MAP = MLE + regularisation from the prior."
- Connection to regularisation: Gaussian prior = L2 (Ridge). Laplace prior = L1 (Lasso). Uniform prior = MAP = MLE.
- "Module 3 preview: when you learn Ridge and Lasso regression, remember — they are Bayesian MAP estimates."
  **Transition**: "Bayesian inference also gives us a predictive distribution."

---

## Slide 37: Bayesian Predictive Distribution (THEORY)

**Time**: ~3 min
**Talking points**:

- "MLE says the next HDB price will be $450K. Bayesian says $450K plus or minus $35K with 95% probability."
- "The predictive distribution captures parameter uncertainty — how unsure you are about mu and sigma."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Now let us learn to make decisions from data with hypothesis testing."

---

## Slide 38: Hypothesis Testing (THEORY)

**Time**: ~4 min
**Talking points**:

- Tell the story: "Are Queenstown HDB prices really higher than Yishun, or is the difference just random noise?"
- H0 (null): no difference. H1 (alternative): there is a difference.
- Walk through the error table: Type I (false alarm), Type II (missed effect).
- "Power = 1 - beta = probability of detecting a real effect."
- If beginners look confused: "Think of a fire alarm. Type I = alarm with no fire. Type II = fire with no alarm. You want to minimise both."
  **Transition**: "The formal framework for this is Neyman-Pearson."

---

## Slide 39: The Neyman-Pearson Framework (THEORY)

**Time**: ~3 min
**Talking points**:

- The likelihood ratio test is the most powerful test at any significance level.
- "A small p-value means the data is surprising under the null hypothesis."
- Critical callout: what the p-value is NOT. "Not the probability that H0 is true. Not the probability of a false positive."
- If beginners look confused: "The p-value says: 'if there really were no difference, how unusual would our result be?'"
  **Transition**: "Before you run an experiment, you need to know how much data you need."

---

## Slide 40: Power Analysis (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the formula and the Singapore example.
- "To detect a $20K price difference between towns, you need about 126 transactions per group."
- FDR correction: "If you test many towns at once, Benjamini-Hochberg controls the false discovery rate."
  **Transition**: "When distributional assumptions fail, there is another approach."

---

## Slide 41: Permutation Tests (ADVANCED)

**Time**: ~3 min
**Talking points**:

- "If there is no real difference, shuffling the labels should not change the result."
- Walk through the code: shuffle 10,000 times, count how often you get a result as extreme.
- "Permutation tests are exact — no distributional assumptions. But they only test the null of exchangeability."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "The bootstrap is the Swiss Army knife of uncertainty estimation."

---

## Slide 42: Bootstrap (THEORY)

**Time**: ~4 min
**Talking points**:

- "Cannot get more data? Pretend you can by resampling with replacement."
- Walk through the code: resample prices 10,000 times, compute median each time, take 2.5th and 97.5th percentiles.
- "This gives you a confidence interval without any distributional assumptions."
- If beginners look confused: "Imagine putting all your data points on slips of paper in a hat. Draw a slip, write down its value, put it BACK, repeat. That is resampling with replacement."
  **Transition**: "The simple bootstrap can be improved."

---

## Slide 43: BCa Bootstrap Intervals (THEORY)

**Time**: ~3 min
**Talking points**:

- "The simple percentile method assumes the bootstrap distribution is symmetric. BCa fixes both bias and skewness."
- When bootstrap fails: extreme quantiles, heavy tails, dependent data, very small n.
- Wild bootstrap for heteroscedastic data — important for HDB data where variance differs by town.

**[CAN SKIP IF RUNNING SHORT]**

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Let us move to one of the most important skills: data visualisation."

---

## Slide 44: Data Visualisation: Seeing Patterns (THEORY)

**Time**: ~3 min
**Talking points**:

- "Anscombe's quartet: four datasets with identical summary statistics but completely different shapes. Only a plot reveals the truth."
- Rule: "Never trust a summary statistic you have not plotted."
- Chart selection guide: distribution = histogram, relationship = scatter, trend = line, comparison = bar, correlation = heatmap, distribution by group = box/violin.
  **Transition**: "Let us build some charts with Plotly."

---

## Slide 45: Plotly: Interactive Charts (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through the histogram and scatter plot code.
- "Plotly charts are interactive — hover, zoom, pan, select. They export to HTML for sharing."

**[SWITCH TO LIVE CODING — 5 min]**

- Demonstrate: histogram of resale prices, scatter plot of area vs price coloured by town.
  **Transition**: "Line charts for trends and heatmaps for correlations."

---

## Slide 46: Plotly: Time Series and Heatmaps (THEORY)

**Time**: ~3 min
**Talking points**:

- Monthly median price by town as a line chart.
- Correlation heatmap: "Values near 0 mean no linear relationship. Check scatter plots for non-linear patterns."

**[SWITCH TO LIVE CODING — 3 min]**

- Demonstrate: line chart of price trends, correlation heatmap.
  **Transition**: "For large data, Polars has an optimisation trick."

---

## Slide 47: Lazy Frames: Processing at Scale (THEORY)

**Time**: ~3 min
**Talking points**:

- Eager vs Lazy: eager runs immediately, lazy builds a query plan and optimises before executing.
- "scan_csv instead of read_csv. Add your operations. Then .collect() runs everything."
- "Polars optimises the query plan: predicate pushdown, projection pushdown. The 15M-row dataset processes in seconds."
- If beginners look confused: "Just know that for large data, use scan_csv instead of read_csv and add .collect() at the end."
  **Transition**: "Now let us automate the entire profiling process."

---

## Slide 48: Automated Data Profiling (THEORY)

**Time**: ~4 min
**Talking points**:

- Walk through what profiling checks: missing values, outliers, duplicates, type mismatches, cardinality, distributions, correlations, imbalance.
- Show the manual profiling code: "This is 30 lines. DataExplorer does it in 3. But you must know what it computes before you trust it."
  **Transition**: "After profiling comes cleaning."

---

## Slide 49: Data Cleaning: The 80% Problem (THEORY)

**Time**: ~3 min
**Talking points**:

- "Data scientists spend 80% of their time cleaning. The other 20% is complaining about it."
- Walk through the six cleaning operations: missing values, duplicates, type coercion, encoding, scaling, outlier handling.
- Show the manual cleaning code.
  **Transition**: "Different types of missing data require different strategies."

---

## Slide 50: Imputation Strategies (THEORY)

**Time**: ~3 min
**Talking points**:

- Three types of missingness: MCAR, MAR, MNAR. The type determines the strategy.
- "Never impute the target variable."
- If beginners look confused: "The key question is: is the data missing for a reason? If yes, dropping rows loses the signal."
  **Transition**: "Now let us see how Kailash automates everything we just learned."

---

## Slide 51: Kailash Bridge: Theory Meets Engine (THEORY)

**Time**: ~3 min
**Talking points**:

- Walk through the three theory-to-engine mappings.
- "Distributions, correlation, outliers = DataExplorer. Type detection, encoding, scaling = PreprocessingPipeline. Histograms, scatter, heatmaps = ModelVisualizer."
- "You learn both paths: manual (to understand) and engine (to be productive)."
  **Transition**: "Let us see DataExplorer in action."

---

## Slide 52: DataExplorer: Automated Profiling

**Time**: ~4 min
**Talking points**:

- Walk through the API: create explorer with alert config, profile the dataset, inspect results.
- 8 alert types: missing data, outliers, high correlation, low variance, class imbalance, duplicates, type mismatch, skewness.
- "DataExplorer implements everything from the profiling section in a single call."
  **Transition**: "You can also compare datasets before and after cleaning."

---

## Slide 53: DataExplorer: Compare Datasets

**Time**: ~3 min
**Talking points**:

- "Did cleaning introduce new problems? Did imputation change the distribution? Always compare before and after."
- "The principle: cleaning should improve data quality without distorting the underlying signal."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "PreprocessingPipeline automates the cleaning itself."

---

## Slide 54: PreprocessingPipeline: Automated Cleaning

**Time**: ~3 min
**Talking points**:

- "Auto-detect types and transform. Three lines instead of thirty."
- Walk through the auto-detection table: numeric = scale, low-cardinality categorical = one-hot, high-cardinality = target encoding, temporal = extract year/month/day.
- "The decisions are the same as manual cleaning. The code is just shorter."
  **Transition**: "ModelVisualizer provides consistent charts."

---

## Slide 55: ModelVisualizer: Consistent Charts

**Time**: ~3 min
**Talking points**:

- "One API for all chart types. Polars-native — no .to_pandas() conversion."
- "In M1, you learn both Plotly directly and ModelVisualizer. Understanding both means you can customise when needed."
  **Transition**: "ConnectionManager stores all your results."

---

## Slide 56: ConnectionManager: Where Results Live

**Time**: ~3 min
**Talking points**:

- "Every profile and transformation is persisted in a database. Reproducibility, comparison, audit trail."
- "Module 3 preview: ConnectionManager becomes central when you persist model results and SHAP values."
  **Transition**: "Let us see how all the engines fit together."

---

## Slide 57: Architecture: How the Engines Fit Together

**Time**: ~3 min
**Talking points**:

- Walk through the table: Load (ASCENTDataLoader) -> Profile (DataExplorer) -> Clean (PreprocessingPipeline) -> Visualise (ModelVisualizer) -> Persist (ConnectionManager).
- "You learn both paths: manual and engine. The manual knowledge lets you debug when the engine gives unexpected results."

**[PAUSE FOR QUESTIONS — 3 min]**

**Transition**: "Let us go deeper into some additional theory before the lab."

---

## Slide 58: Variance: Measuring Spread (THEORY)

**Time**: ~3 min
**Talking points**:

- "The mean tells you where the centre is. Variance tells you how spread out the data is."
- Coefficient of Variation (CV) = sigma/mu. Use it to compare spread across groups with different means.
- Show the Polars code: per-town variance and CV.

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Beyond centre and spread: skewness and kurtosis."

---

## Slide 59: Skewness and Kurtosis (THEORY)

**Time**: ~3 min
**Talking points**:

- "HDB prices are right-skewed. Most are $300K-$600K, but a few reach $1.5M+."
- Kurtosis: "Heavy tails mean more extreme values than the Normal predicts. Financial returns have high kurtosis — that is why Black-Scholes fails."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Pearson correlation only captures linear relationships."

---

## Slide 60: Correlation: Beyond Pearson (THEORY)

**Time**: ~3 min
**Talking points**:

- "Pearson misses non-linear relationships. A perfect parabola has r=0."
- Spearman: rank-based, captures monotonic relationships. Kendall: concordance, good for small samples. Mutual information: captures any dependency (covered in Module 2).

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "The Central Limit Theorem is why hypothesis testing works."

---

## Slide 61: The Central Limit Theorem (THEORY)

**Time**: ~3 min
**Talking points**:

- "Take ANY distribution. Sample from it repeatedly. The average of your samples will be approximately Normal."
- "HDB prices are not Normal. But the average of 100 randomly selected prices IS approximately Normal. This is why confidence intervals work."
- Berry-Esseen: convergence rate is O(1/sqrt(n)).

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "A few more advanced topics, then we head to the lab."

---

## Slides 62-66: Advanced Theory (LLN, BIC/WAIC, FDR, Wild Bootstrap)

**Time**: ~8 min total (or skip entirely)
**Talking points**:

- Law of Large Numbers: sample average converges to population average.
- Bayesian Model Comparison: BIC vs WAIC vs LOO-CV.
- False Discovery Rate: Benjamini-Hochberg procedure.
- Quantiles and Box Plots: IQR, whiskers, outliers.
- Covariance Matrix: foundation for PCA in Module 4.
- Wild Bootstrap: for heteroscedastic data like HDB prices varying by town.

**[CAN SKIP ALL IF RUNNING SHORT]**
**Transition**: "Let us look at some production gotchas before we start the lab."

---

## Slide 67: Simpson's Paradox

**Time**: ~3 min
**Talking points**:

- "Prices appear to be falling overall but rising in every town. How? The mix shifted — more transactions in cheaper towns."
- "Always stratify. An aggregate trend can be the opposite of every subgroup trend."
- Show the Polars code. This is a live demo opportunity.
  **Transition**: "Another common trap."

---

## Slide 68: Survivorship Bias

**Time**: ~2 min
**Talking points**:

- "You only see data that survived. HDB dataset only includes completed transactions — withdrawn listings are invisible."
- "Ask: what data is missing? What selection process created this dataset?"

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Some advanced Polars patterns before the lab."

---

## Slides 69-71: Advanced Polars (when/then, struct, list columns)

**Time**: ~5 min total
**Talking points**:

- when/then/otherwise: conditional columns.
- String operations: extracting year from date strings.
- Expressions as first-class objects: define reusable expressions.
- "Polars expressions are composable. Define once, use everywhere."

**[CAN SKIP IF RUNNING SHORT]**
**Transition**: "Let us load data properly with the shared loader."

---

## Slide 72: ASCENTDataLoader

**Time**: ~2 min
**Talking points**:

- "One loader for all formats. Auto-detects environment — local, Jupyter, Colab."
- "Never hardcode file paths."
- Walk through available M1 datasets: weather, HDB, economic indicators, taxi trips.
  **Transition**: "Now let us start the lab."

**[PAUSE FOR QUESTIONS — 3 min]**

---

## Slide 73: Lab Exercises Overview

**Time**: ~3 min
**Talking points**:

- Walk through the 8 exercises. "Each exercise builds on the previous one. Start with 1.1 and work through."
- "~70% of the code is provided. You fill in the TODO blanks."
- "Solutions are in modules/ascent01/solutions/ — but try the exercise first."

**[SWITCH TO EXERCISE DEMO — 5 min]**

- Open ex_1_1.py. Walk through the structure: header, imports, data loading, TODO markers.
  **Transition**: "For exercises 1.7 and 1.8, you will use the Kailash engines."

---

## Slide 74: Exercise 1.1: Your First Data Exploration

**Time**: ~3 min
**Talking points**:

- Walk through the four tasks: variables, load CSV, inspect, answer a question.
- Show the three format options: local .py, Jupyter, Colab.
- "The exercise is pre-filled. You fill in the TODO blanks."
  **Transition**: "Exercises 1.7 and 1.8 are the capstone — here is what they look like."

---

## Slide 75: Exercises 1.7-1.8: Kailash Engines in Action

**Time**: ~3 min
**Talking points**:

- "1.7: configure DataExplorer alerts, run profiling, interpret the results."
- "1.8: full pipeline — load taxi data, profile, clean, visualise, compare before/after. This is your Module 1 capstone."

**[BEGIN LAB — allocate remaining time]**

---

## Slides 76-78: Discussion Prompts

**Time**: ~10 min total (after lab, or interleaved)
**Talking points**:

- **Missing Data Dilemma**: "12% of remaining_lease is missing, concentrated in pre-1990. Is this MCAR, MAR, or MNAR?" (Answer: MAR — missingness depends on date, which is observed.)
- **Outlier Question**: "200 transactions above $1.2M are real penthouses. Are they errors?" Let the room debate. (Answer: real but extreme. Keep for distribution analysis, possibly remove for modelling.)
- **Correlation Trap**: "Schools correlate with price at r=0.82. Does building schools increase prices?" (Answer: confounding variable is town — mature estates have more schools AND higher prices.)
- **p-value Problem**: "26 tests at alpha=0.05 = 1.3 false positives expected." Bonferroni vs BH.
- **Flash Crash Revisited**: What Polars commands and which DataExplorer alerts would catch it?

**[PAUSE FOR QUESTIONS — 5 min]**

---

## Slides 79-80: Synthesis

**Time**: ~5 min
**Talking points**:

- Walk through the FOUNDATIONS summary: data types, distributions, mean vs median, correlation vs causation, four Polars patterns.
- Walk through the THEORY summary: MLE, Fisher Information, conjugate priors, p-values, bootstrap.
- Walk through the ADVANCED summary: exponential family, information geometry, BIC/WAIC/LOO-CV, wild bootstrap.
- Show the Kailash engine map: DataExplorer, PreprocessingPipeline, ModelVisualizer, ConnectionManager.
  **Transition**: "Next time, Module 2 — Statistical Mastery for ML and AI Success."

---

## Slide 81: Connection to Module 2

**Time**: ~3 min
**Talking points**:

- "You learned to SEE data. Next, you will learn to REASON about it."
- Module 2 adds: FeatureStore, FeatureEngineer, ExperimentTracker.
- "Bayesian thinking, hypothesis testing, causal inference, feature engineering — all building on the foundation you laid today."

---

## Slide 82: Assessment Preview

**Time**: ~2 min
**Talking points**:

- Quiz topics: identify distribution shapes, choose mean vs median, write Polars filter, interpret DataExplorer alerts.
- "AI-resilient: questions require running code and interpreting YOUR specific outputs."
  **Transition**: "That is Module 1 complete. Any final questions?"

**[FINAL Q&A — 5 min]**

---

## Timing Summary

| Section                                        | Slides    | Time                        |
| ---------------------------------------------- | --------- | --------------------------- |
| Title + Intro + Setup                          | 1-5       | ~11 min                     |
| Opening Case (HDB Flash Crash)                 | 6-8       | ~10 min                     |
| Foundations: Python + Polars                   | 9-22      | ~52 min (incl. live coding) |
| Theory Block A: Probability + Distributions    | 23-37     | ~35 min                     |
| Theory Block B: Hypothesis Testing + Bootstrap | 38-43     | ~20 min                     |
| Visualisation + Profiling + Cleaning           | 44-57     | ~40 min (incl. live coding) |
| Additional Theory + Production Gotchas         | 58-72     | ~15 min (most skippable)    |
| Lab Setup + Exercises                          | 73-75     | ~9 min                      |
| Discussion Prompts                             | 76-78     | ~10 min                     |
| Synthesis + Closing                            | 79-82     | ~10 min                     |
| Q&A buffers                                    | scattered | ~13 min                     |
| **Total**                                      |           | **~180 min**                |
