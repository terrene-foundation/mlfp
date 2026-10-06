# Module 1 — Data Pipelines and Visualisation Mastery with Python

> *"Can you trust a number you didn't explore yourself?"*

This chapter is your entry point into the MLFP programme. It starts at zero — no assumed Python, no assumed statistics, no assumed data experience — and takes you to the point where you can run a complete exploratory data analysis pipeline on a messy Singapore dataset using Kailash's data engines. (The course datasets are modelled on public Singapore data but are synthetic or illustrative, with problems planted for you to find; each lesson says which.)

Everything you learn here will be used again. The Polars patterns in Lesson 1.2 will still be on your fingertips in Module 6 when you reshape transformer training logs. The `group_by` / `agg` muscle you build in Lesson 1.3 is the same muscle you'll use to compute per-cohort calibration in Module 5. The visualisation literacy from Lesson 1.6 is what will let you tell the difference between a broken model and a broken chart three modules from now. So do not rush. The chapter is long because the foundations matter.

---

## Learning Outcomes

By the end of this chapter you will be able to:

- Read, write, and execute Python programs that use variables, data types, f-strings, functions, loops, conditionals, imports, and basic error handling — enough Python to be productive in any data-related task.
- Load tabular data from CSV, Parquet, and API sources into Polars DataFrames and inspect their shape, schema, and summary statistics.
- Filter, select, sort, transform, and aggregate datasets of tens of thousands of rows (and far more) using Polars expressions and method chaining, without reaching for pandas.
- Write reusable helper functions that classify values, format numbers, and compute statistics, and apply them through `group_by` / `agg` pipelines.
- Join multiple tables on shared keys, reason about left vs inner vs outer joins, and handle the NULLs that arise after a join.
- Compute rolling averages, year-over-year changes, and rank within group using Polars window functions with `.over()` partitioning.
- Create appropriate, honest, interactive visualisations (histogram, scatter, bar, heatmap, line, stacked bar) using the ModelVisualizer engine and Plotly, and critique charts against Gestalt and Z-pattern reading principles.
- Use the DataExplorer engine to profile a messy dataset automatically, configure AlertConfig thresholds to fit your domain, and interpret each of the eight alert types as a concrete cleaning action.
- Use PreprocessingPipeline to impute missing values, scale numeric columns, and encode categoricals — holding out test rows before fitting, so the prepared train and test data that downstream modules consume carry no leakage.
- Assemble the above into a complete end-to-end pipeline that turns a raw, dirty dataset into a model-ready, auditable report.

Those are the concrete skills. Underneath them sits a more important outcome: you will have learned to *distrust* aggregated numbers you did not inspect yourself, and you will have the hand-tools to inspect them.

---

## Prerequisites

**Formally: none.** This is Lesson zero of the MLFP programme and Lesson zero of your data career. Specifically, we do not assume:

- Prior Python experience (you will learn `print()` on page one).
- Prior statistics (we will define mean, median, variance, and correlation from scratch).
- Prior data experience (we will define what a DataFrame is before using one).
- Prior ML experience (ML does not appear in this chapter — we are building the foundation).

**Practically, you will need:**

- A computer with Python 3.11+ installed, or a Google account so you can run the exercises on Colab. No paid subscriptions are required.
- The `mlfp` repository checked out locally, or the module-1 exercise notebook on Colab.
- The ability to run one command in a terminal: `uv sync` (or `pip install -e .` if you prefer). If that sentence was terrifying, skip the local option and go straight to Colab — everything in this chapter runs there.
- A willingness to type code into a file or a cell and press Enter. You cannot learn programming by reading alone; every worked example in this chapter is designed for you to copy, run, and modify.

If you have seen pandas before, forget it deliberately. Do not translate Polars back into pandas in your head — the mental translation costs you more than the re-learning. Polars is close enough to pandas that the muscle memory transfers, and different enough that fighting it wastes hours.

---

## How to Read This Chapter

This chapter has eight lessons that map one-to-one with the eight exercises in `modules/mlfp01/solutions/`. Each lesson follows the same structure:

1. **Why This Matters** — a Singapore-flavoured story that motivates the lesson. Skip if you are short on time, but the stories are often what the material sticks to in memory.
2. **Core Concepts** — plain-language explanations first, then formal definitions, then code examples, then a "common mistakes" sidebar.
3. **Mathematical Foundations** (where applicable) — the underlying mathematics with derivations. Marked THEORY.
4. **The Kailash Engine** — the engine that implements the lesson's concepts. DataExplorer, PreprocessingPipeline, or ModelVisualizer.
5. **Worked Example** — a complete, step-by-step walkthrough on a course dataset. Every line of code, every output (produced by running the code on the course files), every interpretation.
6. **Try It Yourself** — three to five small drills. Attempt them before reading the answers at the end of each lesson.
7. **Cross-References** — how this lesson connects forward and backward.
8. **Reflection** — what you should now be able to do, and how to verify that.

Within each section, every non-trivial explanation is tagged with one of three depth markers:

| Marker | Audience | How to Read It |
|---|---|---|
| **FOUNDATIONS:** | Zero background | Plain language, analogies, no derivations. Read every word. |
| **THEORY:** | Practitioner | Formal statement, derivation sketch, working knowledge required. Read if you want to reason about why things work. |
| **ADVANCED:** | Masters / researcher | Paper references, frontier results. Skim on first read, return later if interested. |

If you are entirely new to the material, read only the FOUNDATIONS sections on your first pass through a lesson. You will finish Module 1 in roughly 20 hours of reading and 10 hours of exercises, and you will be productive. Come back to the THEORY sections once the FOUNDATIONS have settled — typically one to two weeks later. The ADVANCED material is there to make sure you are not bored if you already have a PhD in statistics.

**Estimated reading time per lesson:**

| Lesson | Title | Reading | Exercise | Total |
|---|---|---|---|---|
| 1.1 | Your First Data Exploration | 90 min | 45 min | ~2h |
| 1.2 | Filtering and Transforming Data | 90 min | 55 min | ~2h 25m |
| 1.3 | Functions and Aggregation | 100 min | 60 min | ~2h 40m |
| 1.4 | Joins and Multi-Table Data | 100 min | 60 min | ~2h 40m |
| 1.5 | Window Functions and Trends | 110 min | 60 min | ~2h 50m |
| 1.6 | Data Visualisation | 100 min | 60 min | ~2h 40m |
| 1.7 | Automated Data Profiling | 110 min | 60 min | ~2h 50m |
| 1.8 | Data Pipelines and End-to-End Project | 120 min | 75 min | ~3h 15m |

Total: roughly 20 hours of focused work. Spread it across two weeks at two hours per day and it is comfortable. Compress it to a weekend and you will forget half of it by Monday.

---

# Lesson 1.1: Your First Data Exploration

## Why This Matters

Consider a scenario. (It is an illustrative scenario, not a reported incident — but every ingredient of it happens in real data teams.) An analytics team publishes a monthly dashboard of the *median* HDB resale price per town. Hidden in the raw transactions is a batch of records with impossible prices: a few hundred sales keyed in at S$10, a few hundred at S$9,000,000 — mis-keyed figures, test rows that leaked into production, a unit error somewhere upstream. The median, which is what the dashboard displays, barely moves. The distribution, which is what almost nobody displays, has two absurd spikes at either end.

Nobody notices. The dashboard looks fine, because a median is *designed* to ignore extremes. Meanwhile every average computed from the same table, and every model later trained on it, quietly learns from the errors. The problem is caught only when somebody opens the raw file, loads it into a DataFrame, and plots a histogram — one line of code.

This is not hypothetical for you. The course's HDB resale dataset (about 50,000 transactions; a synthetic dataset modelled on the public HDB resale data, with data-quality problems planted on purpose) contains exactly this kind of dirt: 107 sales at S$10 and 144 sales at S$9,000,000, while the median sits at about S$849,000. By Lesson 1.2 you will find them yourself.

We will refer to this scenario in this chapter as "the dashboard that said everything was fine". It is the reason the word *mastery* is in the module title. Mastery does not mean that you can write fancy code. It means that you can look at a file of raw data and find things that nobody asked you to look for. It means you do not trust an aggregated number you did not personally plot. It means you know the difference between the mean and the median well enough to explain, unprompted, why property reports use one and not the other — and why that same choice can hide a data error.

In this lesson you will learn the single skill that catches this kind of problem on day one: load a data file into memory, look at the raw data, and compute summary statistics. We will do this on a smaller, friendlier dataset — Singapore monthly weather — so the stakes are low while you are learning the syntax. By the end of Lesson 1.8, the tools in your hands will be the ones that find the planted errors in every course dataset.

## Core Concepts

### FOUNDATIONS: What is a program?

A program is a file of text that tells a computer what to do in a language the computer can understand. You write the text. You run the program. The computer reads your text one line at a time and performs the actions described. If the text contains an instruction the computer does not understand, it stops and prints an error message. That is all a program is. There is no magic.

The language we will use is Python. It is called Python for no particularly important reason (the author liked Monty Python), but the syntax was deliberately designed to read almost like English. You will see the word `if` used for conditional decisions, the word `for` used for loops, and the word `return` used to send a value back from a function. This is a good thing. It means that you can often guess what a line of Python does on your first read, and be right more often than not.

A Python program is just a file with a `.py` extension. You can open it in any text editor, including the one that came with your operating system. The exercises in this course live in files like `ex_1.py`. Inside each file you will find lines of Python that the computer will execute from top to bottom when you run the file. If you prefer notebooks to files, the exercises also exist as `.ipynb` notebooks, which break the same code into runnable cells — but the code inside is identical. Pick the format that makes you comfortable and stick with it.

### FOUNDATIONS: Variables

A variable is a name attached to a value. You create a variable by writing a name, an equals sign, and a value:

```python
city = "Singapore"
```

After this line runs, the word `city` refers to the string `"Singapore"`. You can use the variable anywhere you would use the value itself. If you type `print(city)`, Python will print `Singapore`. If you later reassign the variable — `city = "Jakarta"` — the name now refers to a different value, and the previous one is discarded. Variables are not permanent containers. They are labels you can freely move around.

The name on the left of the `=` sign must follow a few rules. It must start with a letter or underscore. It must contain only letters, digits, and underscores. It cannot be a reserved Python word like `if`, `for`, or `return`. By convention we write variable names in lowercase with underscores between words, like `years_of_data` or `mean_price`. This is called *snake_case*. It is a convention, not a rule, but Python code that does not follow it looks foreign to everyone who reads it.

A variable has a *type* that is determined by the value you assigned to it, not by anything you wrote. Python figures out the type on its own. You do not have to say "this is a string" or "this is a number" — you just assign the value and Python remembers. This is called *dynamic typing*, and it is one of the reasons Python is pleasant to write in.

### FOUNDATIONS: The four types you will actually use in this lesson

Python has many data types built in, but at the start you need only four:

- **`int`** — an integer. Whole numbers, positive or negative: `0`, `1`, `42`, `-3`. No decimal point.
- **`float`** — a floating-point number. Numbers with a decimal point: `1.35`, `-0.01`, `3.141592`. Python uses 64-bit IEEE 754 floats, which means you get about 15 decimal digits of precision. For most data work this is more than enough.
- **`str`** — a string. A sequence of text characters, written inside single or double quotes: `"Singapore"`, `'hello'`, `"123"`. Note that `"123"` is a string, not a number — the quotes make it text.
- **`bool`** — a Boolean. Exactly two possible values: `True` and `False`. The capitalisation matters. Booleans are what you get back from comparisons like `price > 500000`.

You can check the type of any variable by calling `type()` on it:

```python
years_of_data = 30
print(type(years_of_data))   # <class 'int'>

latitude = 1.35
print(type(latitude))         # <class 'float'>

city = "Singapore"
print(type(city))             # <class 'str'>

is_tropical = True
print(type(is_tropical))      # <class 'bool'>
```

Each of those `print` calls writes a line to your terminal. The text in the comments (`#`) is not executed — it is for human readers.

> **Common mistake:** writing `True` as `true`, or `"Singapore"` without the quotes. Python is case-sensitive, and unquoted words are interpreted as variable names. `true` without quotes is an error because Python does not know about a variable called `true`. Similarly, `print(Singapore)` without quotes will raise `NameError: name 'Singapore' is not defined` — Python thinks you are asking it to look up a variable called `Singapore`, which does not exist.

### FOUNDATIONS: Arithmetic

Python does arithmetic with the operators you would expect: `+`, `-`, `*` for multiplication, `/` for division. There are two less obvious ones you will meet immediately:

- `**` raises a number to a power. `2 ** 10` is `1024`.
- `%` is the *modulo* operator — it gives the remainder after division. `17 % 5` is `2`, because 17 divided by 5 is 3 remainder 2. Modulo is useful for asking "is this number divisible by that one?" — a test many data tasks need.

You can freely mix `int` and `float` in arithmetic; the result is a `float` if either operand was a `float`. So `27 + 0.5` is `27.5`, not `27`. Division with `/` always produces a `float` — even `10 / 2` is `5.0`, not `5`. If you want the integer result use `//` (integer division): `10 // 3` is `3`.

Here is a real calculation a property agent would do — the price per square metre of a flat:

```python
resale_price = 485_000     # S$485,000
floor_area_sqm = 93
price_per_sqm = resale_price / floor_area_sqm
print(price_per_sqm)       # 5215.053763440861
```

Notice the underscore in `485_000`. Python lets you insert underscores inside numeric literals as visual separators. It treats `485_000` exactly like `485000` — the underscore is purely for human readability. For large numbers, always use it.

### FOUNDATIONS: f-strings — the way you will build every output line

When you want to print text that includes the current value of a variable, the cleanest way in modern Python is an *f-string*. An f-string is a normal string literal with the letter `f` in front of it. Inside the string, anything between curly braces `{ }` is evaluated as Python and the result is inserted into the text:

```python
price = 485_000
area = 93

print(f"Price: S${price}, Area: {area} sqm")
# Price: S$485000, Area: 93 sqm
```

That is already useful. But f-strings have a format-specifier syntax that makes them powerful for data reporting. After the variable name, you can write a colon and a format code:

```python
price_per_sqm = price / area
print(f"Price per sqm: S${price_per_sqm:,.2f}")
# Price per sqm: S$5,215.05
```

The `:,.2f` is three instructions packed together: the `,` means "insert comma thousands separators", the `.2` means "round to two decimal places", the `f` means "display as a floating-point number". You can use any combination. Other common specifiers you will see throughout the course:

- `:.0f` — zero decimal places, floating-point. `f"{485000.7:.0f}"` gives `"485001"`.
- `:>10` — right-align in a field of width 10. Useful for lining up numbers in a printed table.
- `:<10` — left-align in a field of width 10. Useful for lining up labels in a printed table.
- `:.1%` — display as a percentage with one decimal place. `f"{0.234:.1%}"` gives `"23.4%"`. Note that `0.234` is interpreted as 23.4%, not 0.234%.

You will build every output line in this chapter with f-strings. Get comfortable with them now.

> **Common mistake:** forgetting the `f`. If you write `print("Price: S${price}")` with no `f`, Python will print the literal text `Price: S${price}` instead of substituting the variable. The `f` is the signal that turns a string into an f-string.

### FOUNDATIONS: What is a DataFrame?

Up to this point, everything has been a single value. A variable holds one number or one string. Real data does not look like that. A dataset of Singapore weather has one row for each month and multiple columns — month name, mean temperature, total rainfall. A dataset of HDB resale transactions has one row for each sale and dozens of columns — town, flat type, floor area, lease commencement, resale price. To work with data like this you need a structure that holds a two-dimensional table, and that structure is called a *DataFrame*.

A DataFrame is a rectangular table of data with named columns and rows. Each column has a type (string, integer, float, boolean, date). Each row is an observation. Think of it as a spreadsheet you can manipulate with code instead of a mouse. The DataFrame is the single most important object you will meet in this course. Every piece of data you work with will arrive as one, live inside one, or leave as one.

There are many DataFrame libraries in Python. You may have heard of pandas, the oldest and most widely used. In this course we use *Polars*. Polars is newer, written in Rust, uses less memory, and is considerably faster on large datasets — the course's 50,000-row HDB dataset loads in a fraction of a second, and Polars stays fast well into the tens of millions of rows. More importantly for a learner, Polars has a cleaner, more consistent API than pandas, which means you will make fewer mistakes while you are still new to the vocabulary. Every exercise in MLFP uses Polars.

> **If you have used pandas before:** the mental translation table is short. `pd.read_csv` becomes `pl.read_csv`. `df.loc[df["town"] == "BISHAN"]` becomes `df.filter(pl.col("town") == "BISHAN")`. `df["price"].mean()` is the same in both. The notable differences are that Polars uses `pl.col("name")` to refer to a column inside an expression (you do not slice rows with bracket syntax), and Polars operations like `with_columns` return a new DataFrame rather than modifying in place. There is no `.iloc`, no `inplace=True`, and — blessedly — no `SettingWithCopyWarning`.

A DataFrame has two dimensions: height (number of rows) and width (number of columns). Polars calls these `df.height` and `df.width`, or you can ask for both at once as `df.shape`, which returns a tuple `(height, width)`. A *tuple* is like a list but cannot be modified after creation — you will see this structure often in Python.

Polars also stores *column names* (accessible as `df.columns`) and *column types* (accessible as `df.dtypes`). The column types are Polars types, not Python types: a text column is `pl.String` or `pl.Utf8`, an integer column is `pl.Int64`, a decimal column is `pl.Float64`, a boolean column is `pl.Boolean`, and so on. You do not need to memorise these — they appear in the output of `describe` and other inspection functions, and you can look them up when you see an unfamiliar one.

## Mathematical Foundations

### FOUNDATIONS: The three measures of central tendency

Once you have a column of numbers, the first question you ask is "what is a typical value?" There are three answers, each correct in different circumstances. The mathematics is trivial; the judgement about *when to use which* is not.

**Mean.** The arithmetic mean of a set of numbers is their sum divided by the count:

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

In plain language: add them all up, divide by how many there were. The mean is the most commonly used average because it is easy to compute and has nice mathematical properties — most notably, it is the value $m$ that minimises the sum of squared differences $\sum (x_i - m)^2$. That property will matter enormously when you meet linear regression in Module 2.

The downside of the mean is that it is *sensitive to outliers*. A single extreme value can drag it far from the typical case. If you record the prices of ten HDB flats and nine are around $400,000 and one is a penthouse at $2.5 million, the mean is $610,000 — a number that represents neither the ordinary flats nor the penthouse. It is in the gap between them, where no actual flat lives.

**Median.** The median is the middle value when you sort the numbers. If you have an odd number of values, it is literally the middle one. If you have an even number, it is the average of the two middle values. The median is *robust* — it ignores how extreme the outliers are, only where they are. In the ten-flat example above, the median is around $400,000, which is exactly what "a typical flat" should be.

Singapore property reports use the median for this reason. Even if a single billionaire buys a penthouse for $10 million, the median resale price does not budge. The mean would jump by $50,000 and every news outlet would run a headline about a "housing boom". The median is the honest number.

**Mode.** The mode is the most frequently occurring value. For continuous data like prices it is usually not useful — no two flats sell for exactly the same price. For categorical data like flat type (`3 ROOM`, `4 ROOM`, `5 ROOM`, `EXECUTIVE`), the mode tells you which category is most common, which is exactly what "typical" means for a category.

**When to use which.**

- **Symmetric distribution, no outliers:** mean and median agree; either is fine.
- **Skewed distribution (one long tail):** use the median. This is the default for incomes, prices, and almost any "money" variable.
- **Bimodal distribution (two peaks):** use neither; the single-number summary is misleading. Plot a histogram and describe the two groups separately.
- **Categorical data:** use the mode.

> **Common mistake:** reporting the mean of a skewed distribution as the "average" in a public-facing number. The word "average" in English is ambiguous — some listeners will hear "mean", some will hear "typical value". For skewed data those are different numbers, and using the wrong one is misleading. When in doubt, report the median and call it the median.

### THEORY: Why the mean minimises squared error

Suppose you have numbers $x_1, \dots, x_n$ and you want to pick a single value $m$ that is "closest" to all of them. "Closest" needs a definition; let's use squared distance and add them up:

$$S(m) = \sum_{i=1}^{n} (x_i - m)^2$$

To find the $m$ that minimises this, take the derivative with respect to $m$ and set it to zero:

$$\frac{dS}{dm} = \sum_{i=1}^{n} -2(x_i - m) = -2 \sum_{i=1}^{n} (x_i - m) = 0$$

Divide by $-2$ and split the sum:

$$\sum_{i=1}^{n} x_i - \sum_{i=1}^{n} m = 0 \implies \sum x_i = nm \implies m = \frac{1}{n}\sum x_i = \bar{x}$$

So the mean is the answer. If instead you use absolute distance $|x_i - m|$ and minimise $\sum |x_i - m|$, the answer turns out to be the median — but the derivation involves sub-gradients because absolute value is not differentiable at zero, so we skip it here and return to it in Module 2 when we cover robust regression.

The takeaway: mean = minimises squared loss, median = minimises absolute loss. That pairing is the root of why linear regression (which minimises squared loss) is so sensitive to outliers, and why robust regression methods (which minimise absolute loss or variations thereof) are less so. You will meet these again.

### FOUNDATIONS: Variance and standard deviation

The mean tells you where the centre of a distribution is. It says nothing about how spread out the values are. For spread, the standard tool is the *variance* (symbol $\sigma^2$, read "sigma squared"):

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

In words: take each value's distance from the mean, square it, average the squared distances. Squaring makes every term positive (otherwise positive and negative deviations would cancel) and penalises large deviations more than small ones.

The variance is in squared units. If prices are in dollars, variance is in dollars-squared, which is not a unit anyone has intuition for. So we usually report the *standard deviation* $\sigma$ — the square root of the variance — which is back in the original units:

$$\sigma = \sqrt{\sigma^2}$$

Rough rule of thumb: for roughly bell-shaped data, about 68% of values fall within one standard deviation of the mean, about 95% within two, and about 99.7% within three. In the course weather file, Singapore's monthly mean temperatures have a standard deviation of about 0.6°C — so a month more than about 1.2°C from the annual average would be statistically unusual. For the course HDB resale prices the standard deviation is about S$512,000 — inflated by the planted S$9,000,000 records you will meet in Lesson 1.2, which is itself a lesson: the standard deviation, like the mean, is dragged around by extreme values.

> **Pedantic footnote on $n$ vs $n-1$:** the formula above divides by $n$. Many statistics textbooks and libraries divide by $n-1$ instead. The $n-1$ version is the *sample variance*, which is an unbiased estimator of the *population variance* when you are treating your data as a sample from a larger population. The $n$ version is the *maximum likelihood estimator* of the variance under a normal model. For a dataset with 50,000 HDB transactions, the difference is completely negligible — the two standard deviations differ by about 0.001%. For a dataset with 12 weather records, the variances differ by a factor of 12/11 (about 9%) and the standard deviations by about 4% (0.62 vs 0.60 °C for temperature), which matters. Polars' `.var()` and `.std()` use $n-1$ by default (Bessel's correction); `numpy.var` defaults to $n$. Know which one your library uses. We return to this in Module 2.

### ADVANCED: Robust alternatives to the mean and variance

For heavy-tailed data the mean and variance both break. The standard robust replacements are:

- **Median** for central tendency. Breakdown point 50% — half your data must be contaminated before the median is moved far from the centre.
- **MAD** (Median Absolute Deviation) for spread: $\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$. To get a drop-in replacement for the standard deviation, multiply by the constant 1.4826, which is chosen so that MAD and standard deviation agree for normally distributed data.
- **IQR** (Interquartile Range) — the difference between the 75th and 25th percentiles. Also a common spread measure for skewed data, and the basis for box-plot whiskers.

Robust statistics is a full subfield (see Huber, *Robust Statistics*, 1981). For Module 1 you only need to know they exist and when to reach for them.

## The Kailash Engine: DataExplorer (first look)

This is the engine you will meet formally in Lesson 1.7. In Lesson 1.1 we do not call it yet. Instead we use Polars' own `df.describe()` — a bridge to help you recognise that the per-column computations you just learned (mean, std, min, max) are all available in a single call, which is the first thing DataExplorer automates.

DataExplorer is the Kailash ML engine for automated dataset profiling. It wraps a battery of column-level statistics and quality checks behind a single API. Its full capabilities include:

- Per-column type inference (numeric, categorical, boolean, constant, id, text).
- Summary statistics (count, mean, std, min, max, quartiles, skewness, kurtosis).
- Missing-value counts and percentages.
- Duplicate-row detection.
- Pairwise correlations (Pearson and Spearman) with configurable thresholds.
- Alert generation for eight categories of data quality issues.
- Comparison of two datasets for distribution drift.
- HTML report generation.

You will use all of these in Lesson 1.7. For now, just know that the summary statistics `df.describe()` prints are the starting point of what DataExplorer computes for every numeric column (it adds skewness, outlier counts, correlations and alerts on top). Learning the manual form first is deliberate — when DataExplorer flags something as "high skewness" in Lesson 1.7, you should be able to say "right, that's the mean being different from the median because there's a long tail, I remember that from Lesson 1.1" instead of being confused by a stranger.

## Worked Example: Singapore Monthly Weather

We will work through a complete first exploration of a Singapore weather dataset. This dataset has exactly twelve rows (one per month) and three columns: `month`, `mean_temperature_c`, and `total_rainfall_mm`. The values are monthly climate averages prepared for this course — illustrative, not official station records. It is small on purpose — every output fits on one screen, so you can see everything the code produces. Later lessons use a 50,000-row dataset and you will be ready for it.

If you are running this locally, make sure you have run `uv sync` in the `mlfp` repository once. If you are running on Colab, open the Lesson 1 notebook. Either way, the code below should run as-is; the MLFPDataLoader knows how to find the CSV file in both environments.

### Step 0: Set up the file

Create a new Python file called `lesson_1_1.py` anywhere convenient. At the top, add the imports you will need for the rest of the lesson:

```python
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader
```

The first line, `from __future__ import annotations`, is a Python idiom that makes type hints a little more flexible. You do not need to understand it yet; just include it at the top of every file you write in this course.

The second line imports Polars and gives it the short alias `pl`. Every Polars expression in the course will start with `pl.something`. Typing `polars.something` every time would get tedious, so everyone uses the `pl` alias. This is a convention, not a rule, but it is so universal that you will confuse other readers if you break it.

The third line imports `MLFPDataLoader` from the `shared` helper module that comes with the course. This class knows how to find data files in every environment — local Python, Jupyter notebook, Google Colab — without you having to hardcode any file paths. You instantiate it once, and from then on you call `loader.load("mlfp01", "filename.csv")`.

### Step 1: Load the data

```python
loader = MLFPDataLoader()
df = loader.load("mlfp01", "sg_weather.csv")

print("Data loaded.")
print(df.head(5))
```

When you run this, you should see output something like:

```
Data loaded.
shape: (5, 3)
┌───────────┬─────────────────────┬───────────────────┐
│ month     ┆ mean_temperature_c  ┆ total_rainfall_mm │
│ ---       ┆ ---                 ┆ ---               │
│ str       ┆ f64                 ┆ i64               │
╞═══════════╪═════════════════════╪═══════════════════╡
│ January   ┆ 26.5                ┆ 167               │
│ February  ┆ 27.1                ┆ 108               │
│ March     ┆ 27.5                ┆ 170               │
│ April     ┆ 28.0                ┆ 165               │
│ May       ┆ 28.3                ┆ 171               │
└───────────┴─────────────────────┴───────────────────┘
```

Read the output carefully. At the top, `shape: (5, 3)` tells you that the snippet you printed has 5 rows and 3 columns — that's `head(5)` at work, not the whole dataset. Below that, Polars prints the column names, then the column types on a separate row (`str` for text, `f64` for 64-bit floating-point, `i64` for 64-bit integer — rainfall is recorded in whole millimetres), then the first five rows of data. The Unicode box-drawing characters are Polars' way of making the table readable; they have no meaning beyond that.

The data tells a story already. Mean temperatures in Singapore hover between 26 and 29°C — the equatorial climate that makes people who grew up elsewhere sweat through their first week. Rainfall in these five months sits between about 110 and 170 mm, with February clearly the driest of the five and May the wettest of the five (we only looked at five months, so we can't yet say anything about the whole year).

### Step 2: Inspect the full shape

```python
rows, cols = df.shape
print(f"Rows: {rows}")
print(f"Columns: {cols}")

print("\nColumn names:")
for col_name in df.columns:
    print(f"  - {col_name}")

print("\nColumn types:")
for col_name, dtype in zip(df.columns, df.dtypes):
    print(f"  {col_name}: {dtype}")
```

Two Python concepts appear here that we should name explicitly. First, *tuple unpacking*: the line `rows, cols = df.shape` takes the two-element tuple returned by `df.shape` and assigns the first element to `rows` and the second to `cols`. This is much cleaner than writing `rows = df.shape[0]; cols = df.shape[1]`. Polars, like most Python libraries, follows the convention that shape tuples are `(height, width)`, i.e. rows first then columns.

Second, *for loops*. The `for col_name in df.columns:` line says "for each item in the list `df.columns`, call it `col_name` and run the indented block below". The `df.columns` attribute returns a plain Python list of strings — the column names — so the loop iterates once per column. Inside the loop we print the name. Then `zip(df.columns, df.dtypes)` pairs up the columns list with the types list element-by-element, giving us tuples like `("month", String)`, `("mean_temperature_c", Float64)`, and so on. The second loop unpacks each tuple in the `for` statement itself: `for col_name, dtype in zip(...)`.

You will see both of these patterns — unpacking and `zip` — constantly. They are worth learning once.

Expected output:

```
Rows: 12
Columns: 3

Column names:
  - month
  - mean_temperature_c
  - total_rainfall_mm

Column types:
  month: String
  mean_temperature_c: Float64
  total_rainfall_mm: Int64
```

Twelve rows, exactly one per calendar month. Good: this is what you would expect from a monthly weather dataset. If you saw 11 rows you would immediately know something was missing. The types are as expected — month is text, temperature is a decimal number, and rainfall is a whole number of millimetres.

### Step 3: Summary statistics via describe

```python
print(df.describe())
```

Polars' `.describe()` computes count, null count, mean, standard deviation, min, 25th percentile, 50th percentile (median), 75th percentile, and max for every column. For string columns it computes count, nulls, and nothing else (because mean of strings is meaningless) — except min and max, which for text are alphabetical. Numeric columns are reported as `f64` in the summary, even the integer rainfall column. The output looks like this:

```
shape: (9, 4)
┌────────────┬──────────┬─────────────────────┬───────────────────┐
│ statistic  ┆ month    ┆ mean_temperature_c  ┆ total_rainfall_mm │
│ ---        ┆ ---      ┆ ---                 ┆ ---               │
│ str        ┆ str      ┆ f64                 ┆ f64               │
╞════════════╪══════════╪═════════════════════╪═══════════════════╡
│ count      ┆ 12       ┆ 12.0                ┆ 12.0              │
│ null_count ┆ 0        ┆ 0.0                 ┆ 0.0               │
│ mean       ┆ null     ┆ 27.466667           ┆ 171.75            │
│ std        ┆ null     ┆ 0.624257            ┆ 40.104693         │
│ min        ┆ April    ┆ 26.5                ┆ 108.0             │
│ 25%        ┆ null     ┆ 27.1                ┆ 155.0             │
│ 50%        ┆ null     ┆ 27.6                ┆ 167.0             │
│ 75%        ┆ null     ┆ 27.9                ┆ 171.0             │
│ max        ┆ September┆ 28.3                ┆ 254.0             │
└────────────┴──────────┴─────────────────────┴───────────────────┘
```

This is a staggering amount of information in one call. Let's read it.

Look at `mean_temperature_c` first. Mean 27.47°C, standard deviation 0.62°C, min 26.5°C, max 28.3°C. A standard deviation well under 1°C over the whole year tells you Singapore's climate is extraordinarily stable compared to temperate zones. For comparison, London's monthly mean temperatures range from about 5 to 19°C — a standard deviation closer to 5°C. Singapore is roughly eight times more stable month-to-month.

Look at `total_rainfall_mm`. Mean 172 mm per month, standard deviation 40 mm, min 108 mm, max 254 mm. The wettest month gets well over twice the rain of the driest — monthly rainfall varies a lot, even though the temperature barely changes.

Note that the `min` and `max` rows for the `month` column show `"April"` and `"September"`. For a string column, `describe()` min and max are alphabetical, so these are simply the alphabetically-first and alphabetically-last month names — not the coolest or hottest month, and unrelated to any other column. This is a minor trap: `.describe()` applies its aggregations column-by-column without considering that you probably wanted "which month was coldest", not "which month comes first in the alphabet". We will compute the actually-hottest month in Step 4.

### Step 4: Find the hottest, coldest, and wettest months

To answer "which month was hottest?" you have to filter the DataFrame to keep only the row where `mean_temperature_c` equals its maximum, then read off the month name. Here is the Polars idiom:

```python
max_temp = df["mean_temperature_c"].max()
hottest_row = df.filter(pl.col("mean_temperature_c") == max_temp)
print(hottest_row)
```

Two things to unpack. `df["mean_temperature_c"]` is a Polars Series — a single column extracted from the DataFrame. You can call aggregation methods like `.max()`, `.min()`, `.mean()`, `.std()` directly on a Series, and they return a single scalar value. So `max_temp` after that line is a Python float, the maximum temperature in the dataset.

`df.filter(pl.col("mean_temperature_c") == max_temp)` is the filtering expression. `pl.col("mean_temperature_c")` is the Polars way to refer to a column inside an expression — you are saying "the column called mean_temperature_c". The `==` compares that column to the scalar `max_temp`, producing a Boolean column (True where the temperature equals the max, False everywhere else). `df.filter(...)` keeps only the rows where the Boolean column is True. The result is a DataFrame containing exactly the row(s) with the maximum temperature.

To get just the month name out of that result, index into the `month` column with `[0]`:

```python
hottest_month = hottest_row["month"][0]
hottest_temp = hottest_row["mean_temperature_c"][0]
print(f"Hottest month: {hottest_month} at {hottest_temp:.1f}°C")
```

Expected output:

```
Hottest month: May at 28.3°C
```

The `[0]` is an index into the single-row DataFrame, returning the first (and in this case only) element. Polars Series support Python-style indexing; `series[0]` gives you the first value, `series[-1]` gives you the last.

Repeat the pattern for coldest and wettest:

```python
min_temp = df["mean_temperature_c"].min()
coldest_row = df.filter(pl.col("mean_temperature_c") == min_temp)
coldest_month = coldest_row["month"][0]
coldest_temp = coldest_row["mean_temperature_c"][0]
print(f"Coldest month: {coldest_month} at {coldest_temp:.1f}°C")

max_rain = df["total_rainfall_mm"].max()
wettest_row = df.filter(pl.col("total_rainfall_mm") == max_rain)
wettest_month = wettest_row["month"][0]
wettest_rain = wettest_row["total_rainfall_mm"][0]
print(f"Wettest month: {wettest_month} with {wettest_rain:.1f} mm of rain")
```

Expected output:

```
Coldest month: January at 26.5°C
Wettest month: November with 254.0 mm of rain
```

Look closely at the coldest month. January and December are *tied* at 26.5°C, so `coldest_row` actually contains two rows, and `[0]` silently picks the first one. Always check `coldest_row.height` before reading `[0]` — an extreme value is not always unique. The coolest months (December–January) and the wettest (November–December) cluster at the year's end, which is the northeast monsoon season.

### Step 5: A formatted summary report

The last step is to collect everything into a human-readable report. This is the output a colleague would see. The goal is something you would be comfortable pasting into a Slack channel or an email.

```python
mean_temp = df["mean_temperature_c"].mean()
std_temp = df["mean_temperature_c"].std()
mean_rain = df["total_rainfall_mm"].mean()

separator = "═" * 58

print(f"\n{separator}")
print(f"  SINGAPORE WEATHER SUMMARY")
print(f"{separator}")
print(f"  Total records:   {rows:>6,}")
print(f"  Columns:         {cols:>6}")
print(f"")
print(f"  Temperature (°C)")
print(f"    Mean: {mean_temp:>8.2f}")
print(f"    Std:  {std_temp:>8.2f}")
print(f"    Min:  {min_temp:>8.2f}  ({coldest_month})")
print(f"    Max:  {max_temp:>8.2f}  ({hottest_month})")
print(f"")
print(f"  Rainfall (mm/month)")
print(f"    Mean: {mean_rain:>8.1f}")
print(f"    Max:  {max_rain:>8.1f}  ({wettest_month})")
print(f"{separator}")
```

Output:

```
══════════════════════════════════════════════════════════
  SINGAPORE WEATHER SUMMARY
══════════════════════════════════════════════════════════
  Total records:       12
  Columns:              3

  Temperature (°C)
    Mean:    27.47
    Std:      0.62
    Min:     26.50  (January)
    Max:     28.30  (May)

  Rainfall (mm/month)
    Mean:    171.8
    Max:     254.0  (November)
══════════════════════════════════════════════════════════
```

Two details worth noting. `"═" * 58` is Python string multiplication — it produces a string of 58 copies of the `═` character, giving you a horizontal separator line without having to type it out. You will use this trick for every report you print.

The `:>8.2f` format specifier is what aligns the numbers. `>8` means "right-align in a field eight characters wide", and `.2f` means "two decimal places, float". Right-alignment with a fixed width is what makes numeric columns line up cleanly. Without it, `26.50` and `171.8` would start at different horizontal positions and the report would look messy.

And that is Lesson 1.1 worked end to end. You loaded a dataset, inspected its shape and schema, computed summary statistics both through `.describe()` and through individual column aggregations, filtered to find extreme values, and built a formatted report. Every pattern you just learned will be used again in every subsequent lesson in this chapter.

## Try It Yourself

Before moving to Lesson 1.2, try these five drills. Attempt each one before looking at the answers (which are at the end of this lesson). Resist the urge to copy — typing it yourself is how the muscle memory forms.

**Drill 1.** Write Python that assigns the string `"Orchard Road"` to a variable called `address`, the integer `1998` to a variable called `year_built`, and the float `93.5` to a variable called `area_sqm`. Then print them in a single f-string on one line, formatted as `Orchard Road (built 1998, 93.5 sqm)`.

**Drill 2.** Using the weather DataFrame loaded in Step 1, compute and print the *range* of monthly rainfall — that is, the difference between the maximum and the minimum. Use f-string formatting to display the result to one decimal place with units.

**Drill 3.** Modify the extreme-finding pattern from Step 4 to find and print the month with the *lowest* rainfall in a formatted line that looks like `Driest month: February with 108.0 mm`.

**Drill 4.** What is the coefficient of variation (CV) of monthly rainfall? Recall that CV is defined as $\sigma / \mu$, where $\sigma$ is the standard deviation and $\mu$ is the mean. Report the result as a percentage with one decimal place. Is the CV higher or lower than that of temperature? What does that tell you about the relative variability of rainfall and temperature?

**Drill 5.** Without using `.filter()`, compute the mean temperature of the first six months of the year and the mean temperature of the last six months. Hint: Polars Series support slicing with `df["mean_temperature_c"][:6]` for the first six elements and `df["mean_temperature_c"][6:]` for everything from index 6 onwards.

## Cross-References

- **Lesson 1.2** will pick up from here and teach you to filter by more complex conditions (price ranges, multiple towns, date ranges) and to create new columns from existing ones. The `pl.col()` expression you just met will be everywhere.
- **Lesson 1.6** will visualise the summary statistics you computed in this lesson as histograms, line charts, and heatmaps. The intuition for "skewed vs symmetric" you just built will become visible.
- **Lesson 1.7** will replace much of what you did here with a single call to `DataExplorer.profile()`, but will expect you to understand the per-column statistics — because DataExplorer generates alerts based on them, and alerts are only useful if you can read the underlying numbers.
- **Module 2, Lesson 2.1** will re-derive the mean and variance as maximum-likelihood estimators of a normal distribution. You will see the same formulas from a different angle: not as "what's the average?" but as "what parameter best explains the observed data?".

## Reflection

You should now be able to, without looking anything up:

- Explain what a variable is and what the four basic Python types are.
- Write an f-string that embeds a variable with a specific number format.
- Load a CSV into a Polars DataFrame using `MLFPDataLoader`.
- Call `.shape`, `.columns`, `.dtypes`, `.head()`, and `.describe()` on a DataFrame and describe what each returns.
- Extract a single column as a Series with `df["column_name"]` and call `.mean()`, `.std()`, `.min()`, `.max()` on it.
- Use `.filter(pl.col("column") == value)` to find the row(s) with a specific value.
- Explain the difference between mean and median and name one situation where each is appropriate.
- State, roughly, what fraction of values in a bell-shaped distribution fall within one and two standard deviations of the mean.

If any of those feels shaky, re-read the corresponding section before moving on. Lesson 1.2 assumes all of this is solid.

### Drill answers

1. ```python
   address = "Orchard Road"
   year_built = 1998
   area_sqm = 93.5
   print(f"{address} (built {year_built}, {area_sqm} sqm)")
   ```

2. ```python
   rainfall_range = df["total_rainfall_mm"].max() - df["total_rainfall_mm"].min()
   print(f"Rainfall range: {rainfall_range:.1f} mm")
   ```

3. ```python
   min_rain = df["total_rainfall_mm"].min()
   driest = df.filter(pl.col("total_rainfall_mm") == min_rain)
   print(f"Driest month: {driest['month'][0]} with {driest['total_rainfall_mm'][0]:.1f} mm")
   ```

4. ```python
   cv_rain = df["total_rainfall_mm"].std() / df["total_rainfall_mm"].mean()
   cv_temp = df["mean_temperature_c"].std() / df["mean_temperature_c"].mean()
   print(f"Rainfall CV: {cv_rain:.1%}")
   print(f"Temperature CV: {cv_temp:.1%}")
   ```
   You should see rainfall CV of 23.4% and temperature CV of 2.3%. Rainfall is far more variable than temperature in Singapore — about 10× more variable on a relative basis. This matches intuition: the thermometer barely moves all year but the sky goes from clear to monsoon within days.

5. ```python
   first_half = df["mean_temperature_c"][:6].mean()
   second_half = df["mean_temperature_c"][6:].mean()
   print(f"Jan-Jun mean: {first_half:.2f}°C")
   print(f"Jul-Dec mean: {second_half:.2f}°C")
   ```
   You should see 27.60°C for January–June and 27.33°C for July–December. The first half is slightly warmer because April–June are the hottest months in the file, while the year-end monsoon months (November–December) are among the coolest.

---

# Lesson 1.2: Filtering and Transforming Data

## Why This Matters

The weather dataset in Lesson 1.1 had twelve rows. You could have inspected it with your eyes, no code required. The dataset in this lesson has about fifty thousand rows: HDB resale transactions across ten years (2015–2024). You cannot scroll through fifty thousand rows. If you try, you will miss every pattern of interest. The only way to work with data of this size is to let the computer filter, sort, and transform it while you ask increasingly specific questions.

That is what this lesson is about: asking questions of a dataset that is too large to hold in your head. "Show me only Ang Mo Kio flats." "Of those, show me only the 4-room flats." "Of those, show me only recent transactions and sort them by price." Each of those questions is a filter, a sort, or a transformation — and the cleanest way to stack them together is Polars' method chaining syntax, which you will meet in Step 5 of the worked example.

The habit we are building here is not syntactic. It is interrogative. When you open a new dataset, you should automatically begin asking it questions. What are the extreme values? Which subset looks different from the rest? What does the distribution look like when I cut it by category? The Polars syntax is just the keyboard shortcut for the question. Over time, the questions become automatic; the syntax is the glue that makes them cheap to ask.

## Core Concepts

### FOUNDATIONS: Boolean logic

A Boolean is a value that is either True or False. Every filter you write in this lesson produces a column of Booleans, and the filter operation keeps the rows where that column is True. Before you can filter, you need to know how to write Boolean expressions.

The six *comparison operators* in Python are:

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `==` | equal to | `price == 500_000` | True if price is exactly 500,000 |
| `!=` | not equal to | `town != "BISHAN"` | True if town is anything but BISHAN |
| `>` | greater than | `price > 500_000` | True if price is strictly above 500,000 |
| `<` | less than | `price < 500_000` | True if price is strictly below 500,000 |
| `>=` | greater than or equal to | `price >= 500_000` | True if price is 500,000 or more |
| `<=` | less than or equal to | `price <= 500_000` | True if price is 500,000 or less |

Note the double equals sign `==` for equality. A single `=` means assignment ("store this value in that variable"), which is a totally different operation. Writing `if price = 500_000:` is a syntax error in Python, and that is by design — it prevents the silent bug of accidentally reassigning inside an `if` statement that you meant to be a comparison.

Once you have Boolean values you can combine them with the three logical operators:

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `&` | AND | `(price > 300_000) & (price < 500_000)` | True only if both sides are True |
| `\|` | OR | `(town == "BISHAN") \| (town == "TOA PAYOH")` | True if either side is True |
| `~` | NOT | `~(town == "BISHAN")` | True if the expression inside is False |

**Warning about parentheses.** Polars uses `&`, `|`, and `~` for element-wise Boolean logic on columns. These are the same symbols Python uses for bitwise operations on integers, and they have a different operator precedence from the English `and`, `or`, and `not` keywords. This means you *must* wrap each side of `&` or `|` in parentheses when combining comparisons:

```python
# CORRECT:
df.filter((pl.col("price") > 300_000) & (pl.col("price") < 500_000))

# WRONG — will raise an obscure error:
df.filter(pl.col("price") > 300_000 & pl.col("price") < 500_000)
```

The wrong version is parsed as `pl.col("price") > (300_000 & pl.col("price")) < 500_000`, because `&` binds tighter than `>` in Python. The resulting error is confusing. The rule is simple: every comparison goes in its own pair of parentheses.

> **Forward reference:** you are writing Boolean expressions inside Polars `.filter()` calls. Python also has `if` / `elif` / `else` statements for branching at the level of your code — deciding whether to run one block or another based on a condition. You will meet those in Lesson 1.4. The two are different: Polars filters operate on an entire column of data at once, while `if` statements operate on a single value. Do not mix them up.

### FOUNDATIONS: What `pl.col` actually is

`pl.col("town")` is an *expression*. It is not the column itself — it is a promise to look up the column called `"town"` when Polars eventually evaluates the expression. Expressions are first-class objects in Polars: you can build them up, combine them, and pass them around before they are ever applied to a DataFrame.

This is useful because it means the same expression can be used in many places. `pl.col("price").mean()` is an expression that, when evaluated against a DataFrame, returns the mean of the `price` column. You can use that expression inside `.filter()`, inside `.with_columns()`, inside `.group_by().agg()`, and so on. The expression doesn't care where it is used.

The practical consequence: you will see `pl.col("something")` appear hundreds of times in this chapter. Every time, it is the same idea — a reference to a column. Do not read it as "fetch the column now". Read it as "when the time comes, look up this column".

### FOUNDATIONS: `.filter` — keep rows where a Boolean is True

The `.filter()` method takes a Boolean expression and returns a new DataFrame containing only the rows where the expression is True. It does not modify the original DataFrame. Polars operations are almost always *immutable*: you get a new DataFrame back and the old one is unchanged. This is a deliberate design choice — it eliminates an entire class of bugs where you accidentally mutate data that another part of your code was relying on.

```python
ang_mo_kio = hdb.filter(pl.col("town") == "ANG MO KIO")
```

After this line, `hdb` still contains all 50,150 rows of the original dataset. `ang_mo_kio` is a new DataFrame containing only the Ang Mo Kio rows — 2,486 of them.

You can combine filters in three ways:

1. **Inside a single filter call with `&` / `|`:**
   ```python
   hdb.filter((pl.col("town") == "ANG MO KIO") & (pl.col("resale_price") < 500_000))
   ```
2. **By chaining multiple filter calls:**
   ```python
   hdb.filter(pl.col("town") == "ANG MO KIO").filter(pl.col("resale_price") < 500_000)
   ```
   These two are semantically equivalent. Polars optimises both into the same query plan.
3. **With `.is_in()` for "one of several values":**
   ```python
   central_towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH"]
   hdb.filter(pl.col("town").is_in(central_towns))
   ```
   This is much cleaner than writing `(pl.col("town") == "BISHAN") | (pl.col("town") == "TOA PAYOH") | ...`.

There are also Polars convenience methods for common patterns: `.is_null()` for missing values, `.is_not_null()` for non-missing values, `.str.contains("text")` for substring matching on string columns, `.is_between(low, high)` for closed intervals. You do not need to memorise them — they all start with `pl.col(...)` and read as English-like method chains, so you can often guess the name and look up any that don't work.

### FOUNDATIONS: `.select` — pick columns

`.filter()` is for rows. `.select()` is for columns. It takes the names of the columns you want to keep and returns a new DataFrame with only those columns:

```python
core_cols = hdb.select("month", "town", "flat_type", "floor_area_sqm", "resale_price")
```

The original `hdb` DataFrame has eleven columns; `core_cols` has exactly five. Everything else is dropped. This matters for two reasons. First, it reduces memory: a large dataset takes far less space when it only has five columns instead of eleven. Second, it reduces visual clutter: when you print a DataFrame, only the columns you have selected appear, and you can focus on what matters for the current question.

A rule of thumb: the first thing you usually do when exploring a new question is `.select()` the three to six columns relevant to the question. Working with a wide DataFrame when you only need a narrow one is like writing an email with fifty people on cc when three would do.

### FOUNDATIONS: `.rename` — change column names

Some source datasets have awkward column names — too long, too short, all caps, weird abbreviations. `.rename()` fixes this. You pass a dictionary mapping old names to new names:

```python
renamed = core_cols.rename({
    "month": "sale_month",
    "floor_area_sqm": "area_sqm",
    "resale_price": "price",
})
```

Columns not mentioned in the dictionary keep their old names. The result is a new DataFrame with the renamed columns. Use `.rename()` sparingly — every rename is a potential source of confusion for a reader. But for columns you will reference dozens of times, shortening `resale_price` to `price` can save a lot of typing.

### FOUNDATIONS: `.with_columns` — create new columns

`.with_columns()` adds new columns to a DataFrame (or replaces existing ones). This is how you compute derived values — things that aren't in the raw data but can be calculated from it.

```python
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm")
)
```

Reading this line: take the `resale_price` column, divide it element-wise by the `floor_area_sqm` column, and give the result the name `price_per_sqm`. The `.alias()` method is what names the new column. Without it, the new column would have a generated name like `"literal"` or `"resale_price"` that would be useless.

`price_per_sqm` is a *normalised* measure — it removes the effect of flat size so you can compare two flats fairly. A three-room flat in Bishan might cost $400,000 and a five-room in Jurong might cost $600,000, but the three-room could easily have a higher price per square metre, because size and price both scale with "how desirable is this neighbourhood". Normalising by area lets you ask "which neighbourhood is actually more expensive per unit of space" rather than "which absolute price is bigger".

You can add multiple columns in a single `with_columns()` call by passing multiple expressions:

```python
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)
```

This is more efficient than calling `with_columns` twice in a row, because Polars can fuse the operations into a single pass over the data. For small DataFrames the difference is invisible; for large ones it matters.

The expression `pl.col("month").str.to_date("%Y-%m")` is parsing a string column into a date column. `str.to_date` is a string-namespace method — `pl.col("month").str` gives you access to string functions (`to_date`, `slice`, `contains`, `replace`, `split`, etc.), and `.to_date("%Y-%m")` converts the string into a Polars `Date` using the given format pattern. `%Y` means four-digit year, `%m` means two-digit month. These format codes are standard Python `strftime` conventions, used by almost every date library — they are worth looking up once and bookmarking.

`pl.col("month").str.slice(0, 4)` is taking the first four characters of the string — which in a string like `"2023-01"` gives you `"2023"`. Then `.cast(pl.Int32)` converts that string to an integer. So the new `year` column is an integer — useful for filtering with `pl.col("year") >= 2020`, because integer comparisons are faster and cleaner than string comparisons.

### FOUNDATIONS: `pl.when` — conditional column assignment

Sometimes you want to create a new column whose values depend on a condition. For example, you might want a `price_tier` column that takes the value `"budget"` if the price is below $350k, `"mid_range"` if between $350k and $500k, `"premium"` if between $500k and $700k, and `"luxury"` otherwise. The Polars idiom for this is `pl.when().then().when().then().otherwise()`:

```python
hdb = hdb.with_columns(
    pl.when(pl.col("resale_price") < 350_000)
    .then(pl.lit("budget"))
    .when(pl.col("resale_price") < 500_000)
    .then(pl.lit("mid_range"))
    .when(pl.col("resale_price") < 700_000)
    .then(pl.lit("premium"))
    .otherwise(pl.lit("luxury"))
    .alias("price_tier")
)
```

Reading this: "when the price is less than 350,000, set this cell to 'budget'; otherwise when the price is less than 500,000, set it to 'mid_range'; otherwise when the price is less than 700,000, set it to 'premium'; otherwise set it to 'luxury'. Name the resulting column `price_tier`."

Notice that the `when` clauses are evaluated in order, and each one only applies if the previous ones were False. So the second `when(pl.col("resale_price") < 500_000)` effectively means "between 350,000 and 500,000", because the "less than 350,000" case has already been caught by the first `when`. This is the normal short-circuit behaviour of if/elif chains, and Polars makes it work the same way.

`pl.lit("budget")` wraps the plain Python string `"budget"` into a Polars literal expression. You need `pl.lit` because Polars' `.then()` expects an expression, not a raw Python value, and if you passed the bare string `"budget"` Polars would interpret it as a column reference — and fail, because there is no column called `"budget"`.

### FOUNDATIONS: `.sort` — order rows

`.sort()` orders rows by one or more columns:

```python
hdb.sort("resale_price", descending=True)
```

This returns a new DataFrame sorted from highest to lowest `resale_price`. The default is ascending; `descending=True` flips it. You can sort by multiple columns:

```python
hdb.sort("town", "resale_price", descending=[False, True])
```

This sorts first by `town` alphabetically, then within each town by `resale_price` from highest to lowest. The `descending` parameter can be a single Boolean (applies to all sort columns) or a list of Booleans (one per column).

Sort is an expensive operation — it processes every row. If you are going to sort a filtered subset of the data, filter first and sort second, not the other way around. Polars will often reorder these for you via its query optimiser, but for eager code the order you write matters.

### FOUNDATIONS: Method chaining

Every Polars operation returns a new DataFrame. That means you can stack operations by attaching each one with a dot:

```python
recent_premium = (
    hdb
    .filter(pl.col("year") >= 2020)
    .filter(pl.col("price_tier").is_in(["premium", "luxury"]))
    .select("transaction_date", "town", "flat_type", "price_per_sqm", "resale_price")
    .sort("resale_price", descending=True)
)
```

Read this top-to-bottom: "take the HDB dataset, keep rows from 2020 onwards, keep premium and luxury tiers, pick these five columns, sort by price descending." Each step is a transformation of what came before. The final result is assigned to `recent_premium`. This is called *method chaining*, and it is the idiomatic way to write Polars code.

The parentheses around the whole expression are a Python trick. Inside a pair of parentheses, Python ignores newlines, so you can break a long expression across many lines for readability. Without the parentheses you would have to use backslash line-continuations, which are ugly. The convention: when you have more than two or three chained operations, wrap the whole thing in parentheses and put each `.method()` on its own line.

Method chains are read in the order they appear, and each line is applied to the result of the previous line. Do not try to hold the intermediate DataFrames in your head. Just read linearly: "filter, filter, select, sort". The intermediate states are ephemeral; only the final result matters.

## The Kailash Context

Lesson 1.2 is pure Polars — no Kailash engine is involved yet. But the patterns you are learning here will show up again inside the engines. DataExplorer (Lesson 1.7) uses `pl.col()` expressions internally to compute per-column statistics. PreprocessingPipeline (Lesson 1.8) turns categorical columns into numeric ones — the same kind of column-by-column transformation you are writing by hand with `pl.when().then()`. ModelVisualizer (Lesson 1.6) accepts Polars DataFrames directly as input — you will hand it the output of your `.filter().sort()` chains. Everything you learn about Polars is reusable inside every Kailash engine you will meet.

## Worked Example: HDB Resale Flats

The dataset for this lesson is `hdb_resale.parquet` — 50,150 resale transactions from 2015 to 2024. It is a *synthetic* dataset modelled on the public HDB resale data at `data.gov.sg`: the columns and categories match the public file, but the rows are generated for teaching, prices do not follow the real market's trends, and data-quality problems (such as the S$10 and S$9,000,000 sales) are planted on purpose for you to find. Treat every interpretation in this chapter as a statement about *this file*, not about the Singapore property market. It is also our first use of the *Parquet* file format instead of CSV. Parquet is a columnar storage format that is much more efficient than CSV for numeric data: it is smaller on disk, faster to load, and remembers column types so you do not have to re-parse dates and numbers on every load. Polars reads Parquet with `pl.read_parquet()`, but `MLFPDataLoader.load()` figures out the format from the file extension, so your code is the same.

### Step 1: Load the data

```python
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print(f"Shape: {hdb.shape}")
print(f"Columns: {hdb.columns}")
print(hdb.head(3))
```

Expected output (Polars hides middle columns behind `…` when the table is wider than your terminal):

```
Shape: (50150, 11)
Columns: ['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'remaining_lease', 'resale_price']
shape: (3, 11)
┌─────────┬───────────────┬───────────┬───────┬───┬─────────────────────┬────────────────────┬──────────────┐
│ month   ┆ town          ┆ flat_type ┆ block ┆ … ┆ lease_commence_date ┆ remaining_lease    ┆ resale_price │
│ ---     ┆ ---           ┆ ---       ┆ ---   ┆   ┆ ---                 ┆ ---                ┆ ---          │
│ str     ┆ str           ┆ str       ┆ str   ┆   ┆ i64                 ┆ str                ┆ i64          │
╞═════════╪═══════════════╪═══════════╪═══════╪═══╪═════════════════════╪════════════════════╪══════════════╡
│ 2016-03 ┆ BUKIT PANJANG ┆ 4 ROOM    ┆ 674D  ┆ … ┆ 1996                ┆ 71 years 11 months ┆ 868241       │
│ 2018-04 ┆ TOA PAYOH     ┆ 4 ROOM    ┆ 552B  ┆ … ┆ 2017                ┆ 92                 ┆ 1023539      │
│ 2023-06 ┆ JURONG WEST   ┆ 4 ROOM    ┆ 692   ┆ … ┆ 1975                ┆ 50 years 00 months ┆ 10           │
└─────────┴───────────────┴───────────┴───────┴───┴─────────────────────┴────────────────────┴──────────────┘
```

Fifty thousand rows, eleven columns. That is already too big to scan by eye — and yet three rows are enough to spot two problems. `remaining_lease` is stored as text in two different formats (`"71 years 11 months"` and a bare `"92"`), and the third sale was recorded at S$10. Every subsequent question has to be answered with code, and the first questions should be about whether the data can be trusted.

### Step 2: Basic filters

Start with simple single-condition filters and check the row counts. If a filter returns zero rows, you almost certainly got a column value wrong (wrong capitalisation, wrong spelling, wrong type).

```python
ang_mo_kio = hdb.filter(pl.col("town") == "ANG MO KIO")
print(f"Ang Mo Kio transactions: {ang_mo_kio.height:,}")

four_room = hdb.filter(pl.col("flat_type") == "4 ROOM")
print(f"4-room flats: {four_room.height:,}")

affordable = hdb.filter(
    (pl.col("resale_price") >= 300_000) & (pl.col("resale_price") <= 500_000)
)
print(f"Transactions S$300k-500k: {affordable.height:,}")
```

Expected output:

```
Ang Mo Kio transactions: 2,486
4-room flats: 20,299
Transactions S$300k-500k: 2,885
```

Only 2,885 of 50,150 sales (under 6%) fall between S$300k and S$500k. In this dataset the typical sale is far above that band — the median is about S$849,000.

Notice that the `town` values are all-caps. This is a quirk of the source data — HDB publishes town names in all capitals. If you had written `pl.col("town") == "Ang Mo Kio"` (title case), you would have got zero rows and spent ten minutes wondering why. When a filter returns zero, always inspect the actual values in the column with `df["town"].unique()` to see what casing the source data uses.

### Step 3: Combined filters

Now combine three conditions into a single filter:

```python
amk_4room_affordable = hdb.filter(
    (pl.col("town") == "ANG MO KIO")
    & (pl.col("flat_type") == "4 ROOM")
    & (pl.col("resale_price") <= 500_000)
)
print(f"AMK 4-room under S$500k: {amk_4room_affordable.height:,}")
```

Expected output:

```
AMK 4-room under S$500k: 0
```

Zero rows. Earlier we said a zero-row result usually means a typo — so check before you conclude anything. `"ANG MO KIO"` and `"4 ROOM"` both matched rows on their own (2,486 and 20,299), so the spelling is fine. Drop the price condition and look at the prices instead: the 1,015 Ang Mo Kio 4-room sales range from S$618,796 upwards, with a median of about S$827,000. The answer really is zero — in this dataset no Ang Mo Kio 4-room flat sold for S$500k or less. Telling "my filter is wrong" apart from "the answer is genuinely empty" is exactly the interrogative habit this lesson is about.

Each `&` narrows the result further. The combined filter must be a subset of each single filter — `amk_4room_affordable` can never have more rows than `ang_mo_kio` (2,486), `four_room` (20,299) or `affordable` (2,885). The intersection is never larger than any of its parts, and as here it can be empty.

And use `.is_in()` for the central-towns question:

```python
central_towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH"]
central = hdb.filter(pl.col("town").is_in(central_towns))
print(f"Central towns transactions: {central.height:,}")
# Central towns transactions: 6,495
```

This is cleaner than `(pl.col("town") == "BISHAN") | (pl.col("town") == "TOA PAYOH") | ...`.

### Step 4: Select and rename

Narrow the DataFrame to the columns you actually need, then clean up the names:

```python
core_cols = hdb.select(
    "month", "town", "flat_type", "floor_area_sqm", "resale_price"
)
renamed = core_cols.rename({
    "month": "sale_month",
    "floor_area_sqm": "area_sqm",
    "resale_price": "price",
})
print(renamed.head(3))
```

Expected:

```
shape: (3, 5)
┌────────────┬───────────────┬───────────┬──────────┬─────────┐
│ sale_month ┆ town          ┆ flat_type ┆ area_sqm ┆ price   │
│ ---        ┆ ---           ┆ ---       ┆ ---      ┆ ---     │
│ str        ┆ str           ┆ str       ┆ f64      ┆ i64     │
╞════════════╪═══════════════╪═══════════╪══════════╪═════════╡
│ 2016-03    ┆ BUKIT PANJANG ┆ 4 ROOM    ┆ 95.7     ┆ 868241  │
│ 2018-04    ┆ TOA PAYOH     ┆ 4 ROOM    ┆ 104.5    ┆ 1023539 │
│ 2023-06    ┆ JURONG WEST   ┆ 4 ROOM    ┆ 95.4     ┆ 10      │
└────────────┴───────────────┴───────────┴──────────┴─────────┘
```

Now when you write `pl.col("price")` you do not have to remember whether it was `resale_price` or `resale_price_sgd` or something else; it is just `price`.

### Step 5: Derive new columns

Add `price_per_sqm`, `transaction_date`, and `year` to the original `hdb` DataFrame:

```python
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)

print(hdb.select("month", "transaction_date", "year", "price_per_sqm").head(5))
```

Expected:

```
shape: (5, 4)
┌─────────┬──────────────────┬──────┬───────────────┐
│ month   ┆ transaction_date ┆ year ┆ price_per_sqm │
│ ---     ┆ ---              ┆ ---  ┆ ---           │
│ str     ┆ date             ┆ i32  ┆ f64           │
╞═════════╪══════════════════╪══════╪═══════════════╡
│ 2016-03 ┆ 2016-03-01       ┆ 2016 ┆ 9072.528736   │
│ 2018-04 ┆ 2018-04-01       ┆ 2018 ┆ 9794.631579   │
│ 2023-06 ┆ 2023-06-01       ┆ 2023 ┆ 0.104822      │
│ 2022-11 ┆ 2022-11-01       ┆ 2022 ┆ 9300.893054   │
│ 2021-11 ┆ 2021-11-01       ┆ 2021 ┆ 9506.095552   │
└─────────┴──────────────────┴──────┴───────────────┘
```

Most rows sit around S$9,000–9,800 per square metre. The third row, at S$0.10 per square metre, is the S$10 sale you saw in Step 1 — a derived column inherits every error in the columns it was computed from. Note also that `transaction_date` is a real `date` and `year` is an integer (`i32`), so both can be compared and sorted numerically.

### Step 6: Conditional price tier

Add the `price_tier` column with `pl.when().then()`:

```python
hdb = hdb.with_columns(
    pl.when(pl.col("resale_price") < 350_000).then(pl.lit("budget"))
    .when(pl.col("resale_price") < 500_000).then(pl.lit("mid_range"))
    .when(pl.col("resale_price") < 700_000).then(pl.lit("premium"))
    .otherwise(pl.lit("luxury"))
    .alias("price_tier")
)

tier_counts = (
    hdb.group_by("price_tier")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print(tier_counts)
```

(We'll cover `group_by` in Lesson 1.3; here we use it as a quick way to see how many rows ended up in each tier.) Expected output:

```
shape: (4, 2)
┌────────────┬───────┐
│ price_tier ┆ count │
│ ---        ┆ ---   │
│ str        ┆ u32   │
╞════════════╪═══════╡
│ luxury     ┆ 35157 │
│ premium    ┆ 11664 │
│ mid_range  ┆ 2160  │
│ budget     ┆ 1169  │
└────────────┴───────┘
```

The "luxury" tier holds 35,157 of 50,150 sales — 70%. Budget and mid-range together are under 7%. A tier scheme where most rows are "luxury" is not telling you much: the S$350k / S$500k / S$700k cut-offs were chosen without looking at this data, whose 25th, 50th and 75th percentiles are about S$656k, S$849k and S$1.02M (27.5% of sales are S$1M or more). The lesson is general — before you hard-code thresholds, check the quantiles (`hdb["resale_price"].quantile(0.25)` and friends), or the categories you build will be lopsided. Note also that the 107 planted S$10 sales land in "budget": a tier column built on dirty data is dirty too.

### Step 7: Chain everything together

The real payoff is when you chain filters, selects, and sorts into a single readable pipeline:

```python
recent_premium = (
    hdb
    .filter(pl.col("year") >= 2020)
    .filter(pl.col("price_tier").is_in(["premium", "luxury"]))
    .select(
        "transaction_date", "town", "flat_type",
        "price_per_sqm", "price_tier", "resale_price"
    )
    .sort("resale_price", descending=True)
)

print(f"Count: {recent_premium.height:,}")
print(recent_premium.head(10))
```

Reading the chain top-to-bottom: "start with the HDB dataset, keep rows from 2020 onwards, keep premium and luxury tiers, pick these six columns, sort by price descending". It returns 23,254 rows, and the output starts with five sales at exactly S$9,000,000 (with prices per square metre near S$95,000). Those are the planted bad records again. A sort to the top is one of the fastest ways to surface outliers: the first rows of a descending sort should always be read with suspicion before they are reported.

The `recent_premium` DataFrame is the answer to a specific question ("what are the highest-priced recent HDB resales?"), derived from the raw data in five lines of chained Polars. If someone asks the next question ("now group them by town"), you add another line to the chain. This is the rhythm of exploratory data analysis: one question, one chain, one answer, next question.

## Try It Yourself

**Drill 1.** Filter `hdb` to keep only 4-room or 5-room flats (use `.is_in()`) in Bishan that sold for more than $600,000. How many rows does the result contain?

**Drill 2.** Create a new column `price_per_year_lease` that is `resale_price / lease_commence_date`. The result will be meaningless (dividing price by a year gives you nonsense units) but the exercise is about the mechanics of `with_columns` and `.alias`. Print the first five rows of the resulting column along with `resale_price` and `lease_commence_date`.

**Drill 3.** Write a single chained expression that: filters to 2023 transactions only, adds a `price_per_sqm` column, selects `town`, `flat_type`, `floor_area_sqm`, `resale_price`, `price_per_sqm`, and sorts by `price_per_sqm` descending. Print the top 10.

**Drill 4.** Use `pl.when().then()` to create a column `flat_size_category` where floor_area_sqm ≤ 60 is `"compact"`, 60–90 is `"standard"`, 90–120 is `"large"`, and above 120 is `"jumbo"`. Count how many rows fall into each category.

**Drill 5.** What is the median `price_per_sqm` in the `luxury` tier? In the `budget` tier? What is the ratio between them? (Hint: filter to the tier, then call `.median()` on the column.)

## Cross-References

- **Lesson 1.3** will use the `group_by` + `agg` pattern that appeared briefly in Step 6 of this lesson, and combine it with functions and loops for reusable analysis.
- **Lesson 1.5** will extend `.with_columns` with window functions like `rolling_mean` and `shift`, which compute values across *nearby rows* rather than per-row.
- **Lesson 1.8** will use Polars filtering extensively to clean a messy taxi dataset — removing negative fares, impossible passenger counts, trips dated in the future, and duplicate trip IDs.

## Reflection

You should now be able to:

- Explain the difference between `&` / `|` / `~` in Polars and why comparison expressions must be parenthesised.
- Write a Polars filter that combines two or more conditions.
- Use `.select()` to narrow a DataFrame to a few columns and `.rename()` to clean the names.
- Use `.with_columns()` with `.alias()` to add derived columns, including columns built from arithmetic on existing columns.
- Use `pl.when().then().otherwise()` to build a conditional column with three or more branches.
- Sort a DataFrame by one or more columns in ascending or descending order.
- Write a multi-step method-chain pipeline that combines filter, select, and sort.

### Drill answers

1. ```python
   result = hdb.filter(
       pl.col("flat_type").is_in(["4 ROOM", "5 ROOM"]) &
       (pl.col("town") == "BISHAN") &
       (pl.col("resale_price") > 600_000)
   )
   print(result.height)   # 956
   ```
2. ```python
   hdb.with_columns(
       (pl.col("resale_price") / pl.col("lease_commence_date")).alias("price_per_year_lease")
   ).select("resale_price", "lease_commence_date", "price_per_year_lease").head(5)
   ```
3. ```python
   top_psm = (
       hdb.filter(pl.col("year") == 2023)
       .with_columns((pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"))
       .select("town", "flat_type", "floor_area_sqm", "resale_price", "price_per_sqm")
       .sort("price_per_sqm", descending=True)
       .head(10)
   )
   ```
4. ```python
   hdb = hdb.with_columns(
       pl.when(pl.col("floor_area_sqm") <= 60).then(pl.lit("compact"))
       .when(pl.col("floor_area_sqm") <= 90).then(pl.lit("standard"))
       .when(pl.col("floor_area_sqm") <= 120).then(pl.lit("large"))
       .otherwise(pl.lit("jumbo"))
       .alias("flat_size_category")
   )
   print(hdb.group_by("flat_size_category").agg(pl.len().alias("count")))
   ```
   You should see large 25,696, standard 12,636, jumbo 9,835 and compact 1,983 (the row order of a `group_by` result is not guaranteed — add `.sort("count", descending=True)` if you want it stable).
5. ```python
   lux_psm = hdb.filter(pl.col("price_tier") == "luxury")["price_per_sqm"].median()
   bud_psm = hdb.filter(pl.col("price_tier") == "budget")["price_per_sqm"].median()
   print(f"Luxury median PSM: {lux_psm:.0f}")
   print(f"Budget median PSM: {bud_psm:.0f}")
   print(f"Ratio: {lux_psm / bud_psm:.2f}")
   ```
   You should see roughly S$8,938 (luxury) and S$7,698 (budget), a ratio of about 1.16. The luxury tier costs far more in absolute terms, but only about 16% more per square metre — its median flat is 103 sqm against 40 sqm for the budget tier. In this dataset, the price tiers mostly separate *big* flats from *small* ones. Normalising by area is what reveals that.

---

# Lesson 1.3: Functions and Aggregation

## Why This Matters

So far every question you have asked has been a one-off: filter for this town, compute that mean, print the result. Real data analysis is not one-off. You ask the same question twenty-seven times — once for each HDB town in the dataset — and compare the answers. You want to know the median price of every district, not just Ang Mo Kio. You want the standard deviation too, and the 75th percentile, and the transaction count, all at once. Writing twenty-seven separate filter-and-compute blocks would be six hundred lines of code and the inevitable bug from copy-pasting.

The antidote is two-fold. First, *functions* let you package a calculation under a name so it can be reused. Write the calculation once, give it a name, call it as many times as you want. Second, *group-by aggregation* lets you tell Polars "for each distinct value in this column, compute these statistics" in a single call. You do not loop over the groups yourself; Polars does it for you, and does it fast.

By the end of this lesson you will have written your first Python functions, used them to classify and format values, and aggregated a 50,000-row dataset into a twenty-seven-row district summary table. The district table is the kind of output that ends up on slide three of a property-market briefing — except you will have built it yourself, which means you will know exactly how every number was computed and will not be caught out when someone asks.

## Core Concepts

### FOUNDATIONS: What is a function?

A function is a named block of code that takes inputs, does something with them, and optionally returns a value. You write the function once, and you call it by name as many times as you want. Every time you call it, the inputs can be different, but the computation is the same.

Defining a function uses the `def` keyword:

```python
def format_sgd(amount):
    return f"S${amount:,.0f}"
```

Reading this line by line: `def format_sgd(amount):` says "I am defining a function called `format_sgd` that takes one input parameter, called `amount`". The colon at the end starts a block. Every indented line after the colon is part of the function body — Python uses indentation to define blocks, not curly braces as some other languages do. The `return` statement sends a value back to the caller; when you call `format_sgd(485_000)` you get back the string `"S$485,000"`.

To call a function, write its name followed by parentheses containing the arguments:

```python
print(format_sgd(485_000))       # S$485,000
print(format_sgd(1_200_000))     # S$1,200,000
```

The parameter `amount` inside the function is a local variable — it only exists while the function is running. The caller does not see it. You can give the function a different name when you call it, and you can call it from anywhere — even from inside another function.

Functions are important for two reasons. First, they prevent repetition: if you format a price in Singapore dollars fifty times in a report, you want that formatting logic in one place so that when you decide to change the currency symbol or add a decimal point, you change it once instead of fifty times. Second, they hide complexity: the caller only needs to know what goes in and what comes out; the details of how the function computes its result are not the caller's problem. When you see `compute_iqr(series)` in code, you do not need to look inside to guess what it does — the name tells you it computes the interquartile range.

### FOUNDATIONS: Parameters and return values

Functions can take any number of parameters (including zero):

```python
def greet():
    return "Hello, Singapore!"

def add(a, b):
    return a + b

def describe_flat(town, flat_type, price, area):
    price_per_sqm = price / area
    return f"{flat_type} in {town}: S${price:,.0f} ({price_per_sqm:,.0f}/sqm)"
```

The last one shows that you can do arbitrary computation inside the function and return a formatted result. Every parameter is a local variable inside the function; the caller has no way to see or modify them except through what the function returns.

You can give parameters *default values* so the caller can omit them:

```python
def format_price(amount, currency="SGD"):
    return f"{currency}${amount:,.0f}"

print(format_price(485_000))           # SGD$485,000
print(format_price(485_000, "USD"))    # USD$485,000
```

In the second call we override the default. In the first, `currency` takes its default value. Default parameters make functions more flexible without cluttering the common case.

### THEORY: Type hints

Python is a dynamically typed language — functions accept any type at runtime — but you can *annotate* parameter and return types to document what the function expects:

```python
def format_sgd(amount: float) -> str:
    return f"S${amount:,.0f}"
```

The `: float` after `amount` is a type hint saying "this parameter should be a float". The `-> str` after the closing parenthesis is a return-type hint saying "this function returns a string". Python itself does not enforce these hints at runtime — you can still pass an `int` to `format_sgd` and it will work (ints are formatted fine by the f-string). The hints are documentation for humans and for static type checkers like `mypy`, which can flag incorrect usage before you run the code.

Throughout MLFP we use type hints on all functions. It is a good habit. When you come back to your own code a month later, the hints tell you what the function expects without having to re-read the body.

### FOUNDATIONS: `if` / `elif` / `else`

Inside a function you will frequently want to do one thing in some cases and a different thing in others. Python's conditional statement is `if` / `elif` / `else`:

```python
def price_range_label(price: float) -> str:
    if price < 350_000:
        return "Budget (<350k)"
    elif price < 500_000:
        return "Mid-range (350k-500k)"
    elif price < 700_000:
        return "Premium (500k-700k)"
    else:
        return "Luxury (700k+)"
```

Read this as: "if the price is less than 350,000, return 'Budget'. Otherwise, if the price is less than 500,000, return 'Mid-range'. Otherwise, if the price is less than 700,000, return 'Premium'. Otherwise, return 'Luxury'." The branches are evaluated top-to-bottom, and only the first branch whose condition is True is executed. `elif` is short for "else if" and lets you chain multiple conditions without deeply nesting indentation.

This is the Python-code equivalent of `pl.when().then()` from Lesson 1.2. The difference: `if`/`elif`/`else` operates on a single value at a time (one flat's price), while `pl.when().then()` operates on an entire column at once. When you are writing a function that takes one value and returns one value, use Python `if`. When you are building a column based on another column, use `pl.when`. Both exist because both are needed.

> **Common mistake:** using `if` inside a Polars filter. Writing `hdb.filter(if pl.col("price") > 500_000)` is a syntax error — `if` is a statement, not an expression. The Polars equivalent is `hdb.filter(pl.col("price") > 500_000)` — a Boolean expression that evaluates element-wise.

### FOUNDATIONS: Lists, dictionaries, and loops

Before you can call a function on many values, you need a way to hold many values. Python has two fundamental collection types you will use constantly:

**Lists** are ordered, mutable sequences. You create a list with square brackets:

```python
towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "ANG MO KIO"]
print(towns[0])      # BISHAN
print(towns[-1])     # ANG MO KIO (negative indices count from the end)
print(len(towns))    # 4
towns.append("BEDOK")
print(towns)         # ['BISHAN', 'TOA PAYOH', 'QUEENSTOWN', 'ANG MO KIO', 'BEDOK']
```

You can iterate over a list with a `for` loop:

```python
for town in towns:
    print(town)
```

This runs the indented block once for each item in the list, with the variable `town` bound to that item each time. For loops work on any *iterable* — lists, strings, dictionaries, Polars Series, even the rows of a DataFrame.

**Dictionaries** are collections of key-value pairs. Since Python 3.7 they remember insertion order — iterating gives the keys in the order you added them. You create a dictionary with curly braces:

```python
prices = {"BISHAN": 580_000, "QUEENSTOWN": 620_000, "YISHUN": 420_000}
print(prices["BISHAN"])    # 580000
prices["JURONG WEST"] = 400_000   # add a new key
print(len(prices))         # 4
```

Dictionaries are the tool you use when you need to look up a value by a key — town name to mean price, flat type to count, any mapping from one kind of thing to another. When you iterate over a dictionary with a `for` loop, you get the keys by default. To get both keys and values use `.items()`:

```python
for town, price in prices.items():
    print(f"{town}: S${price:,}")
```

### FOUNDATIONS: Iterating over DataFrame rows

Occasionally — not often, but sometimes — you want to process each row of a DataFrame one at a time in Python. Polars provides `.iter_rows()` for this. The `named=True` option yields each row as a dictionary, which is much more readable than a tuple:

```python
for row in top_10_districts.iter_rows(named=True):
    town = row["town"]
    median = row["median_price"]
    print(f"{town}: S${median:,.0f}")
```

This is fine for formatting output — building a printed report one line per row. It is *not* fine for computation. Iterating over rows in Python to compute something is typically 100–1000× slower than the vectorised Polars equivalent. Rule of thumb: if you are doing arithmetic inside the loop, you should be using a Polars expression instead. If you are just formatting strings, `iter_rows` is the right tool.

### FOUNDATIONS: `group_by` + `agg` — the most important pattern in data analysis

Here is the question: "for each town, what is the median HDB resale price?" In SQL this would be `SELECT town, MEDIAN(resale_price) FROM hdb GROUP BY town`. In Polars it is:

```python
hdb.group_by("town").agg(pl.col("resale_price").median().alias("median_price"))
```

The pattern is always the same:

1. `.group_by(col)` — split the DataFrame into groups, one per unique value of `col`.
2. `.agg(expressions)` — compute one or more aggregation expressions for each group.

The result is a new DataFrame with one row per group, where the first column is the grouping key and the rest are the aggregated values.

You can pass multiple aggregation expressions to `.agg()` to compute many statistics at once:

```python
district_stats = (
    hdb.group_by("town")
    .agg(
        pl.len().alias("transaction_count"),
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").min().alias("min_price"),
        pl.col("resale_price").max().alias("max_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
    )
    .sort("median_price", descending=True)
)
```

This single call processes 50,150 rows and produces a 27-row summary table (one row per HDB town in the dataset). It does count, mean, median, standard deviation, min, max, and two quantiles — eight statistics per town — in a single pass. On modern hardware this completes in well under a second. Trying to do the same thing with a manual for loop over the towns would take minutes and dozens of lines of code.

`pl.len()` is a special aggregation that returns the number of rows in each group — the "count" column. It is different from `pl.col("something").count()`, which counts non-null values in a specific column. When you want "how many transactions in this group", use `pl.len()`.

You can group by multiple columns at once:

```python
town_flat_stats = (
    hdb.group_by("town", "flat_type")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
    )
)
```

This produces one row per unique `(town, flat_type)` pair. With 27 towns and 6 flat types, the result could have up to 162 rows. In the course dataset every combination occurs, so you get all 162; in real data some combinations would be missing (not every town has multi-generation flats), and a multi-key group_by only returns the pairs that actually occur.

### THEORY: What group_by does under the hood

Internally, group_by performs the following steps:

1. **Hash the group keys.** For each row, compute a hash of the group-key columns. Rows with the same hash go into the same bucket.
2. **Partition the data.** Distribute rows across worker threads using the hash. Polars parallelises group_by operations across all cores.
3. **Aggregate within each group.** For each bucket, compute the aggregation expressions. Single-pass aggregations like `mean`, `count`, `sum`, `min`, and `max` are *online* — they update a running state as they scan rows, so memory use is O(number of groups), not O(number of rows).
4. **Collect the results.** Gather the per-group results into a single output DataFrame, one row per group.

For some aggregations, like `median` and `quantile`, the online approach doesn't work — you need the full distribution to compute a quantile. These aggregations are slightly slower because they require buffering the values in each group. Polars handles the difference transparently.

The takeaway: `group_by` is fast because it is parallelised and because most aggregations are streaming. You do not need to optimise it or write it differently for large data. Just use it.

## The Kailash Context

Kailash's engines use `group_by` extensively under the hood. When DataExplorer profiles a categorical column it effectively does a `group_by` to count each category. When PreprocessingPipeline encodes a categorical column with target encoding it does a `group_by` on the category and averages the target value per group. When ModelVisualizer produces a box plot per district it does a `group_by` on the district and computes quartiles. You are learning the pattern that the engines are built on. That is the reason this lesson is here: so that you understand what the engines are doing, not just how to call them.

## Worked Example: District Statistics

We continue with the HDB dataset. The goal of the worked example is to build a complete district-level report — one row per town — with counts, price statistics, and derived spread metrics, then iterate over the top 15 rows to print a formatted report.

### Step 1: Set up and define helper functions

Start with the imports and a couple of helper functions. Functions that format numbers or classify values are common enough that you will write them every day; practise now.

```python
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def format_sgd(amount: float) -> str:
    """Format a number as Singapore dollars with thousands separator."""
    return f"S${amount:,.0f}"


def price_range_label(price: float) -> str:
    """Classify a resale price into a human-readable tier."""
    if price < 350_000:
        return "Budget (<350k)"
    elif price < 500_000:
        return "Mid-range (350k-500k)"
    elif price < 700_000:
        return "Premium (500k-700k)"
    else:
        return "Luxury (700k+)"


def compute_iqr(series: pl.Series) -> float:
    """Compute the interquartile range (Q3 - Q1) of a Polars Series."""
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    if q75 is None or q25 is None:
        return 0.0
    return q75 - q25
```

Three functions. The first formats a number as SGD. The second classifies a price into a tier. The third computes an interquartile range — the difference between the 75th and 25th percentiles — which is a robust measure of spread that ignores the extreme tails.

Note the `"""..."""` strings on the first line of each function body. These are *docstrings* — documentation for the function that can be read by Python's `help()` function, IDE tooltips, and documentation generators. Every function you write in this course should have a one-line docstring saying what it does. It is a small investment that pays back every time you read your own code a week later.

Test the functions before using them:

```python
print(format_sgd(485_000))                                      # S$485,000
print(price_range_label(485_000))                               # Mid-range (350k-500k)
print(price_range_label(720_000))                               # Luxury (700k+)

test_prices = pl.Series("prices", [300_000, 400_000, 500_000, 600_000, 700_000])
print(f"IQR of test prices: {format_sgd(compute_iqr(test_prices))}")  # S$200,000
```

Always test helpers on a small example before plugging them into a 50,000-row pipeline. If the IQR helper returns nonsense for five values, it will return nonsense for fifty thousand.

### Step 2: The district statistics group_by

Load the data and add the derived columns we need:

```python
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
)
```

Then run the big group_by:

```python
district_stats = (
    hdb.group_by("town")
    .agg(
        pl.len().alias("transaction_count"),
        pl.col("resale_price").mean().alias("mean_price"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("resale_price").std().alias("std_price"),
        pl.col("resale_price").min().alias("min_price"),
        pl.col("resale_price").max().alias("max_price"),
        pl.col("resale_price").quantile(0.25).alias("q25_price"),
        pl.col("resale_price").quantile(0.75).alias("q75_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.col("floor_area_sqm").median().alias("median_area_sqm"),
    )
    .sort("median_price", descending=True)
)

print(f"Districts: {district_stats.height}")
print(district_stats.head(5))
```

The output is a tidy table of 27 rows sorted from most expensive to least, with ten columns of statistics per row (plus the town). In the course dataset the top three are TOA PAYOH (median S$973,887), KALLANG/WHAMPOA and BUKIT TIMAH, and the last is BOON LAY (S$819,981). Look at the gap: seven central towns have medians around S$945k–975k, and the other twenty sit tightly between S$820k and S$843k. The synthetic data was generated with a central-town premium and very little else.

### Step 3: Add derived columns to the aggregated table

Now that you have the per-town statistics, you can compute further derived columns the same way as on any DataFrame:

```python
district_stats = district_stats.with_columns(
    (pl.col("q75_price") - pl.col("q25_price")).alias("iqr_price"),
    (pl.col("std_price") / pl.col("mean_price") * 100).alias("cv_price_pct"),
    (pl.col("median_price") / pl.col("max_price")).alias("premium_ratio"),
)
```

Three new columns:

- `iqr_price` — the interquartile range. A wide IQR means prices vary a lot within the town (diverse housing stock).
- `cv_price_pct` — the *coefficient of variation* as a percentage. CV is standard deviation divided by mean, expressed as a percentage. Unlike the raw std, CV is scale-invariant: a CV of 25% means "the spread is a quarter of the average", regardless of whether the average is $400k or $4,000. This makes CV the right tool for comparing spread across groups with different means.
- `premium_ratio` — the median divided by the max. In principle, a ratio near 1 means the median is close to the maximum (the whole town is expensive) and a ratio near 0.5 means a wide range. In practice, look at what you get: about 0.09–0.11 for 26 of the 27 towns, because almost every town contains one of the planted S$9,000,000 sales, so the "max" is that bad record. Only JURONG EAST, whose maximum is a plausible S$1.76M, shows 0.48. A statistic built on the max is a statistic built on the single most extreme — and most likely erroneous — row. The same planted records inflate `cv_price_pct` (37%–78% across towns), because the standard deviation is as sensitive to outliers as the max. The IQR (about S$340k–465k per town) is the robust spread measure here.

Print the core columns:

```python
print(district_stats.select(
    "town", "transaction_count", "median_price", "iqr_price", "cv_price_pct"
).head(10))
```

### Step 4: Multi-key group_by — (town, flat_type) combinations

Sometimes one grouping key is not enough. "Median price per town" is useful but hides the flat type — a town with lots of five-room flats will look more expensive than a town with mostly three-room, even if the price per square metre is the same. To control for flat type, group by both columns:

```python
town_flat_stats = (
    hdb.group_by("town", "flat_type")
    .agg(
        pl.len().alias("count"),
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("town", "flat_type")
)

print(town_flat_stats.filter(pl.col("town") == "ANG MO KIO"))
```

Expected:

```
shape: (6, 5)
┌────────────┬──────────────────┬───────┬──────────────┬──────────────────┐
│ town       ┆ flat_type        ┆ count ┆ median_price ┆ median_price_sqm │
│ ---        ┆ ---              ┆ ---   ┆ ---          ┆ ---              │
│ str        ┆ str              ┆ u32   ┆ f64          ┆ f64              │
╞════════════╪══════════════════╪═══════╪══════════════╪══════════════════╡
│ ANG MO KIO ┆ 2 ROOM           ┆ 102   ┆ 338943.0     ┆ 8366.161398      │
│ ANG MO KIO ┆ 3 ROOM           ┆ 613   ┆ 575598.0     ┆ 8477.134588      │
│ ANG MO KIO ┆ 4 ROOM           ┆ 1015  ┆ 826574.0     ┆ 8482.880658      │
│ ANG MO KIO ┆ 5 ROOM           ┆ 521   ┆ 1.061327e6   ┆ 8905.741351      │
│ ANG MO KIO ┆ EXECUTIVE        ┆ 213   ┆ 1.289661e6   ┆ 8825.950069      │
│ ANG MO KIO ┆ MULTI-GENERATION ┆ 22    ┆ 1526430.5    ┆ 8818.088804      │
└────────────┴──────────────────┴───────┴──────────────┴──────────────────┘
```

(Polars switches to scientific notation such as `1.061327e6` for some large floats — that is 1,061,327.)

Now you can see the structure: within a single town, the median price climbs steadily with flat size, from about S$339k for a 2-room to about S$1.53M for a multi-generation flat. The `median_price_sqm` column tells the opposite story: it barely moves (S$8,366 to S$8,906). Bigger flats cost more *because they are bigger*, not because each square metre is dearer. That is exactly why you normalise by area before comparing towns with different flat mixes.

### Step 5: Iterate over the district report and format each line

Write a helper function that formats one row of the district stats as a report line, then loop over the top 15 rows and call the helper for each:

```python
def district_report_line(row: dict) -> str:
    """Format one district row as a human-readable report line."""
    town = row["town"]
    median = format_sgd(row["median_price"])
    count = row["transaction_count"]
    cv = row["cv_price_pct"]
    sqm = format_sgd(row["median_price_sqm"])
    return f"  {town:<20} {median:>12}  {count:>8,}  CV={cv:5.1f}%  {sqm:>12}/sqm"


print(f"\n{'=' * 70}")
print(f"  SINGAPORE HDB DISTRICT PRICE REPORT")
print(f"{'=' * 70}")
print(f"  {'Town':<20} {'Median Price':>12}  {'Txns':>8}  {'Spread':>8}  {'Per sqm':>12}")
print(f"  {'-' * 66}")

top_15 = district_stats.head(15)
for row in top_15.iter_rows(named=True):
    print(district_report_line(row))

print(f"{'=' * 70}")
```

Expected output (first five of the fifteen lines):

```
======================================================================
  SINGAPORE HDB DISTRICT PRICE REPORT
======================================================================
  Town                 Median Price      Txns    Spread       Per sqm
  ------------------------------------------------------------------
  TOA PAYOH               S$973,887     1,463  CV= 37.1%       S$9,972/sqm
  KALLANG/WHAMPOA         S$964,675     1,514  CV= 47.3%       S$9,960/sqm
  BUKIT TIMAH             S$961,182       485  CV= 47.6%       S$9,809/sqm
  QUEENSTOWN              S$960,422     1,523  CV= 46.5%       S$9,947/sqm
  CENTRAL AREA            S$955,722       504  CV= 68.1%       S$9,938/sqm
  ...
======================================================================
```

The most expensive towns by median price, all central, all near S$9,800–10,000 per square metre against about S$8,600 for the rest. The CV column varies a lot (37% to 68% here) — but remember from Step 3 that it is inflated by the planted S$9M sales, so a high CV here mostly says "this town contains a bad record", not "this town has diverse housing". Before you interpret spread, clean the data (Lesson 1.7) or use the IQR.

### Step 6: Cross-district summary

One more pass — a summary of the summary:

```python
all_medians = district_stats["median_price"]
print(f"\nCross-district summary:")
print(f"  Most expensive district:   {format_sgd(all_medians.max())}")
print(f"  Least expensive district:  {format_sgd(all_medians.min())}")
print(f"  Average district median:   {format_sgd(all_medians.mean())}")
print(f"  Price spread (max - min):  {format_sgd(all_medians.max() - all_medians.min())}")
```

You should see S$973,887 (most expensive), S$819,980 (least), S$865,923 (average district median) and a spread of S$153,906. The spread is the difference between the most and least expensive town medians. Here it is about 18% of the typical price — location matters, but in this synthetic dataset the town effect is a simple central-versus-other step rather than a smooth gradient.

Notice we wrote `format_sgd` five times. Because it is a function, changing the formatting in one place (say, to add a decimal point) updates every output line. That is the payoff for writing helper functions.

## Try It Yourself

**Drill 1.** Write a function `format_percent(value: float, decimals: int = 1) -> str` that formats a decimal between 0 and 1 as a percentage string with the given number of decimal places. For example, `format_percent(0.2345)` should return `"23.5%"` and `format_percent(0.2345, 2)` should return `"23.45%"`.

**Drill 2.** Group the HDB data by `flat_type` and compute the count, median price, and median price per sqm per flat type. Sort by median price ascending. How many distinct flat types are there? Which is most common by transaction count?

**Drill 3.** Write a function `growth_category(yoy_pct: float) -> str` that returns `"declining"` if the value is below -1, `"flat"` if between -1 and 1, `"growing"` if between 1 and 5, and `"booming"` if above 5.

**Drill 4.** Compute a two-level group_by: for each (year, flat_type) combination, the median price. Filter the result to 4-room flats only and print it sorted by year. Is there a clear upward trend? (You need the `year` column created in Step 2.)

**Drill 5.** Using `iter_rows(named=True)`, write a for loop that prints the town names of the five districts with the *widest* spread (highest `iqr_price`). Use a helper function that formats each line.

## Cross-References

- **Lesson 1.4** will join the HDB data to MRT station and school data, letting you ask questions like "do towns closer to the city centre, or with more MRT stations, cost more?"
- **Lesson 1.5** will replace `group_by` with *window functions* for calculations that need to keep the row-level detail, such as rolling averages and YoY changes per town.
- **Lesson 1.6** will visualise the district statistics as a bar chart ranked by median price — the same data, communicated visually.
- **Module 2** will use `group_by` for feature engineering: computing per-user aggregates, per-category means for target encoding, and per-cohort statistics for feature stores.

## Reflection

You should now be able to:

- Define a Python function with parameters, a return value, a docstring, and type hints.
- Use `if` / `elif` / `else` inside a function to return different values based on the input.
- Create and iterate over lists and dictionaries.
- Use `df.group_by(col).agg(expressions)` to compute per-group statistics on a DataFrame.
- Pass multiple aggregation expressions to a single `.agg()` call to compute many statistics at once.
- Group by multiple columns to build cross-tabulations.
- Use `iter_rows(named=True)` to iterate over DataFrame rows in Python and feed each row into a helper function.
- Explain the difference between `pl.when` (column-wise conditional) and Python `if` (value-wise conditional) and know when to use each.

### Drill answers

1. ```python
   def format_percent(value: float, decimals: int = 1) -> str:
       return f"{value * 100:.{decimals}f}%"
   ```
2. ```python
   flat_stats = (
       hdb.group_by("flat_type")
       .agg(
           pl.len().alias("count"),
           pl.col("resale_price").median().alias("median_price"),
           pl.col("price_per_sqm").median().alias("median_price_sqm"),
       )
       .sort("median_price")
   )
   print(flat_stats)
   ```
   There are 6 flat types: 2 ROOM, 3 ROOM, 4 ROOM, 5 ROOM, EXECUTIVE and MULTI-GENERATION. 4 ROOM is the most common with 20,299 transactions. Sorted by median price, the order runs from 2 ROOM (about S$343k) to MULTI-GENERATION (about S$1.49M).
3. ```python
   def growth_category(yoy_pct: float) -> str:
       if yoy_pct < -1:
           return "declining"
       elif yoy_pct < 1:
           return "flat"
       elif yoy_pct < 5:
           return "growing"
       else:
           return "booming"
   ```
4. ```python
   yf = (
       hdb.group_by("year", "flat_type")
       .agg(pl.col("resale_price").median().alias("median_price"))
       .filter(pl.col("flat_type") == "4 ROOM")
       .sort("year")
   )
   print(yf)
   ```
   No. The 4-room median stays between about S$843k and S$852k in every year from 2015 to 2024 — a flat line. The real resale market rose over this decade, but this synthetic dataset was generated without a time trend. Check the data before you repeat a story you already believe about it.
5. ```python
   widest = district_stats.sort("iqr_price", descending=True).head(5)
   for row in widest.iter_rows(named=True):
       print(f"  {row['town']:<20}  IQR={format_sgd(row['iqr_price'])}")
   ```

---

# Lesson 1.4: Joins and Multi-Table Data

## Why This Matters

Real questions rarely fit inside a single table. "Do towns with more MRT stations command higher prices?" requires the HDB transactions table plus a table of MRT stations. "Do districts with more schools command higher prices?" requires the HDB table plus a schools table. "What is the correlation between a district's median price and its distance to the CBD?" requires the HDB table plus station locations from which to compute that distance.

The operation that combines two tables on a shared key is called a *join*. Joins are the workhorse of relational data work; every SQL database has them, every DataFrame library has them, and the same mental model — match rows based on a shared key, decide what to do with the non-matches — applies everywhere. In this lesson you will learn what a join is, the four types you actually need (left, inner, right, outer), how to choose between them, and how to handle the nulls that arise when a join misses.

You will also meet your first `import` of a module from Python's standard library (`math`, used to compute distances), and your first `if` statement in pure Python (as distinct from `pl.when` inside a Polars expression). These are small additions to the Python vocabulary but they unlock a lot.

## Core Concepts

### FOUNDATIONS: What is a join?

A join combines two tables by matching rows that share a common *key*. Suppose you have two tables:

**Table A: HDB transactions** (key column: `town`)
```
town        flat_type   price
BISHAN      4 ROOM      580000
BISHAN      5 ROOM      720000
YISHUN      4 ROOM      420000
PUNGGOL     4 ROOM      485000
```

**Table B: town MRT data** (key column: `town`)
```
town        station_count   km_to_cbd
BISHAN      5               7.9
YISHUN      2               15.7
PUNGGOL     3               13.5
SEMBAWANG   3               18.3
```

(These two small tables are made up to illustrate the idea. The worked example uses the real course files.)

A *join on `town`* produces a new table where rows from A and B that share the same `town` value are combined into a single row:

```
town        flat_type   price    station_count   km_to_cbd
BISHAN      4 ROOM      580000   5               7.9
BISHAN      5 ROOM      720000   5               7.9
YISHUN      4 ROOM      420000   2               15.7
PUNGGOL     4 ROOM      485000   3               13.5
```

Each row from Table A is augmented with the matching columns from Table B. If Table A has multiple rows with the same key (as BISHAN does above), each of them gets the same right-hand columns. The reverse is the dangerous case: if Table B had *two* BISHAN rows, each Bishan sale would be copied twice — a join multiplies rows whenever the key is not unique on the right. You will hit exactly this in the worked example. If Table A has a key that does not appear in Table B, or vice versa, what happens depends on the *join type*.

### FOUNDATIONS: The four join types

There are four standard join types. They differ only in what happens to non-matching rows.

**Inner join.** Keep only rows where the key exists in *both* tables. Rows with no match on either side are dropped. In our example, an inner join of HDB with MRT drops the SEMBAWANG row from Table B (because there are no HDB transactions for Sembawang in our example A) and would drop any HDB row whose town does not appear in Table B.

Use an inner join when you *require* the data from both sides — for example, "give me only HDB transactions for towns where we have MRT data". If a town has no MRT data, you cannot answer the question for that town, so dropping is the right choice.

**Left join.** Keep all rows from the left (first) table. For rows where the key is not found in the right table, fill the right-hand columns with NULL. In our example, a left join of HDB with MRT keeps every HDB row. If there were an HDB row with town "JURONG ISLAND" (a real edge case — there are no HDB flats on Jurong Island, but bear with the hypothetical) and no matching row in Table B, the resulting row would have `station_count = null` and `km_to_cbd = null`.

Use a left join when the left table is the "primary" dataset and the right table is "enrichment" — you want to add information where available without losing any original rows. This is by far the most common join type in practice.

**Right join.** Mirror image of left join: keep all rows from the right table, fill left-hand columns with NULL where needed. Polars supports right joins, but in practice you almost never use them; you can always rewrite `A.join(B, how="right")` as `B.join(A, how="left")`, which is usually clearer.

**Outer (full) join.** Keep all rows from both tables. Rows with no match on one side get NULLs on that side. In our example, an outer join keeps every HDB row *and* every MRT row, including SEMBAWANG (with null flat_type and null price). Use an outer join when you want the union of the two datasets — for example, when you are reconciling two sources and want to see what is in each but not the other.

> **Rule of thumb:** 90% of the time you want a left join. 9% of the time you want an inner join. The remaining 1% is split between right and outer joins. If you cannot articulate why you are using anything other than left, default to left and check the null counts afterward.

### FOUNDATIONS: Polars `.join` syntax

The Polars method is `.join()`. The basic form is:

```python
enriched = hdb.join(mrt_stations, on="town", how="left")
```

- The *left* table is the one you call `.join` on (`hdb`).
- The *right* table is the argument (`mrt_stations`).
- `on="town"` says "match rows where the `town` column is equal in both tables".
- `how="left"` says "keep all rows from the left table, fill missing right-side values with NULL".

If the key column has different names in the two tables, use `left_on` and `right_on`:

```python
hdb.join(mrt_stations, left_on="town", right_on="area_name", how="left")
```

If the tables share multiple key columns, pass a list:

```python
hdb.join(monthly_stats, on=["year", "town"], how="left")
```

After a join, the resulting DataFrame has all the columns from the left table plus the columns from the right table (except the join key, which is not duplicated — for `how="full"`, pass `coalesce=True` to get the same single key column).

Joining does not check that the key values are spelled the same way, and it does not check that the right-hand key is unique. Both are your job — that is what the next two sections are for. If both tables happen to have a column with the same name other than the join key, Polars will rename the right-side version with a suffix (default: `_right`), or you can specify your own suffix with the `suffix=` parameter.

### FOUNDATIONS: Always check the null count after a left join

A left join never drops rows, but it can silently introduce NULLs wherever a match was missing. Always check how many nulls appeared in the right-side columns after a left join:

```python
enriched = hdb.join(mrt_stations, on="town", how="left")
print(f"Nulls in station_count: {enriched['station_count'].null_count()}")
```

If the null count is zero, every row matched and you can proceed. If the null count is non-trivial (say, more than 1% of the rows), you have a choice to make:

1. **Fill the nulls with a sensible default** — for example, `.fill_null(0)` for a count column. Be careful: a fill value is a claim about the world ("0 stations"), and it is wrong whenever the null only means "not found in the other table".
2. **Drop the rows** — if the downstream analysis cannot cope with nulls and the missing data are a small fraction.
3. **Investigate** — the nulls might indicate a data pipeline bug where the right-side dataset is incomplete or the join keys are misaligned (e.g., case mismatch: `"BISHAN"` vs `"Bishan"`).

Option 3 should always come first. Silent null-filling can hide bugs that bite you much later.

### FOUNDATIONS: Predicting nulls with set operations

Before you join, you can predict exactly how many nulls you will introduce by comparing the distinct values of the join key in both tables. Python's built-in `set` type is perfect for this:

```python
hdb_towns = set(hdb["town"].unique().to_list())
mrt_towns = set(mrt_stations["town"].unique().to_list())

matched = hdb_towns & mrt_towns     # intersection: towns in BOTH
unmatched = hdb_towns - mrt_towns   # difference: towns in HDB but not MRT

print(f"HDB towns: {len(hdb_towns)}")
print(f"MRT towns: {len(mrt_towns)}")
print(f"Matched (will join): {len(matched)}")
print(f"Unmatched (will be NULL after left join): {len(unmatched)}")
if unmatched:
    print(f"  Unmatched: {sorted(unmatched)}")
```

The `&` operator on two Python sets gives the intersection (elements in both). The `-` operator gives the difference (elements in the first but not the second). If `unmatched` is non-empty, you know exactly which towns will have NULL after the left join and can decide what to do about them before you run the join.

This is a healthy habit. It takes three extra lines of code and catches the kind of bug where a typo in a town name silently discards 10% of your data.

### FOUNDATIONS: `import` and packages

Until now every import statement you have seen has pulled something from a file that came with the course (`shared`) or from Polars. Python's import system is more general than that: you can import from any installed package. Installed packages live in what Python calls *site-packages* and are usually installed with a tool like `pip` or `uv`.

A handful of imports you will see in this chapter:

```python
import polars as pl         # alias pl for polars
import numpy as np          # alias np for numpy (used in Lesson 1.6)
from datetime import date   # import a specific name from a module
```

The three forms are:

- `import x` — make the entire module `x` available as `x.something`.
- `import x as y` — same but with a different name.
- `from x import y` — pull just `y` out of module `x` into the current namespace.

Use `import x as y` for top-level libraries (polars, numpy, pandas — the `pl`, `np`, `pd` aliases are universal). Use `from x import y` for specific classes and functions you want to use by name repeatedly.

### THEORY: Join as a set-theoretic operation

Formally, a join is a restricted version of a *Cartesian product*. The Cartesian product of two tables with $m$ and $n$ rows produces $m \times n$ rows — every combination. A join filters the Cartesian product down to the rows where the join predicate holds. For an equi-join (join where the predicate is "key columns are equal"), the output has one row for every matching pair.

Inner join: $A \bowtie B = \{(a, b) : a \in A, b \in B, a.\text{key} = b.\text{key}\}$.

Left join: inner join $\cup$ $\{(a, \text{null}) : a \in A, \forall b \in B: a.\text{key} \ne b.\text{key}\}$. That is, the inner-join results plus one row per unmatched left row, with NULL on the right.

Outer join: $A \bowtie_{\text{left}} B \cup B \bowtie_{\text{left}} A$, deduplicated. The symmetric case.

Modern databases (and Polars) do not compute the Cartesian product literally; they use hash joins or sort-merge joins for $O(m + n)$ performance instead of $O(mn)$. But the set-theoretic semantics are still how you should reason about what the output will look like.

## The Kailash Context

The joins you will do in this lesson produce an enriched HDB table with town-level context (MRT station count, distance to the CBD, school count). In Module 2 you will meet `FeatureStore`, the Kailash engine for versioning and retrieving feature sets. Feature stores are essentially long-lived joined tables: you compute your enriched dataset once, version it, store it, and retrieve it during training and serving. The join patterns you are learning here are exactly the ones a feature store applies under the hood when it materialises a feature group. This is why you are learning them first — everything that comes later is a specialised form of what you can do manually today.

## Worked Example: Enriching HDB with MRT and School Data

### Step 1: Load three tables

```python
from __future__ import annotations

import math

import polars as pl

from shared import MLFPDataLoader

loader = MLFPDataLoader()

hdb = loader.load("mlfp01", "hdb_resale.parquet")
mrt_stations = loader.load("mlfp_assessment", "mrt_stations.parquet")
schools = loader.load("mlfp_assessment", "schools.parquet")

print(f"HDB: {hdb.shape}")
print(f"MRT: {mrt_stations.shape}")
print(f"Schools: {schools.shape}")
```

Expected output:

```text
HDB: (50150, 11)
MRT: (150, 7)
Schools: (242, 4)
```

Before any code, ask what one row of each table *is* — its grain. One HDB row is one sale. One MRT row is one **station** (150 stations spread over 32 towns, so most towns have several rows), with columns `station_name, town, line, latitude, longitude, nearest_mrt, distance_to_mrt_km`. One schools row is one school (`school_name, town, type, zone`). Neither right-hand table is one row per town, so neither can be joined onto the HDB table as it stands.

Read the column meanings too. `distance_to_mrt_km` sounds like "how far a flat is from the MRT", but it is the distance from each *station* to its nearest *neighbouring* station (Bukit Batok to Bukit Gombak is 1.145 km). The HDB table has no flat coordinates, so the distance from a flat to its MRT station cannot be computed from this data at all. What we *can* build honestly are town-level features: how many stations a town has, the typical spacing between them, and where the town's stations sit — which gives each town's distance to the city centre.

### Step 2: Inspect each table before joining

Never join blindly. Look at each table first — and especially at the join key:

```python
print(hdb["town"].unique().sort().head(3).to_list())
print(mrt_stations["town"].unique().sort().head(3).to_list())
print(schools["town"].unique().sort().head(3).to_list())
```

```text
['ANG MO KIO', 'BEDOK', 'BISHAN']
['Ang Mo Kio', 'Bedok', 'Bishan']
['Ang Mo Kio', 'Bedok', 'Bishan']
```

The HDB table uses `"ANG MO KIO"` in capitals; the MRT and schools tables use `"Ang Mo Kio"` in title case. To a computer these are different strings. Joined as they are, *nothing* would match.

### Step 3: Predict nulls

```python
hdb_towns = set(hdb["town"].unique().to_list())
mrt_towns = set(mrt_stations["town"].unique().to_list())
print(f"Matched as-is: {len(hdb_towns & mrt_towns)}")

mrt_towns_upper = {t.upper() for t in mrt_towns}
matched = hdb_towns & mrt_towns_upper
unmatched = hdb_towns - mrt_towns_upper

print(f"HDB towns: {len(hdb_towns)}")
print(f"MRT towns: {len(mrt_towns_upper)}")
print(f"Matched after upper-casing: {len(matched)}")
print(f"Unmatched: {sorted(unmatched)}")
```

Expected output:

```text
Matched as-is: 0
HDB towns: 27
MRT towns: 32
Matched after upper-casing: 21
Unmatched: ['BOON LAY', 'CENTRAL AREA', 'HOUGANG', 'KALLANG/WHAMPOA', 'PUNGGOL', 'SENGKANG']
```

As-is, zero towns match: a left join would run without error and give you 50,150 rows with every MRT column null. Upper-casing fixes 21 of the 27 towns. The other six genuinely have no entry in the station table under that name (the station table uses its own town names, such as `Kallang` and `Downtown`, which do not line up with HDB's `KALLANG/WHAMPOA` and `CENTRAL AREA`). You now know, before joining, exactly which towns will be null.

### Step 4: Fix the grain, then the left join

First, see what happens if you fix the casing but forget the grain:

```python
mrt_upper = mrt_stations.with_columns(pl.col("town").str.to_uppercase())
naive = hdb.join(mrt_upper.select("town", "station_name"), on="town", how="left")
print(f"HDB rows: {hdb.height:,}  ->  after naive join: {naive.height:,}")
```

```text
HDB rows: 50,150  ->  after naive join: 186,997
```

The left join did not "preserve the left row count" — it multiplied it by 3.7. Every sale in a town with five stations was copied five times, once per station. This is the many-to-many trap: a left join preserves row count *only* when the key is unique on the right-hand side. Any average you computed from `naive` would silently weight towns by their number of stations.

The fix is to aggregate the station table to one row per town *before* joining:

```python
mrt_by_town = (
    mrt_upper.group_by("town")
    .agg(
        pl.len().alias("station_count"),
        pl.col("distance_to_mrt_km").median().alias("station_spacing_km"),
        pl.col("latitude").mean().alias("town_lat"),
        pl.col("longitude").mean().alias("town_lng"),
    )
)
print(f"MRT by town: {mrt_by_town.shape}")

hdb_enriched = hdb.join(mrt_by_town, on="town", how="left")
print(f"Rows after join: {hdb_enriched.height:,}")
```

```text
MRT by town: (32, 5)
Rows after join: 50,150
```

Now the right-hand key is unique (32 rows, 32 towns), and the row count is preserved. The four new columns are honest town-level features: `station_count` (MRT access), `station_spacing_km` (how far apart the town's stations are), and `town_lat` / `town_lng` (the centre of the town's stations, used in Step 8). Note that we did not bring `nearest_mrt` across — it names a station that may be in a different town, and it means nothing at town level.

### Step 5: Pre-aggregate the schools table

The schools table has the same two problems — title-case towns and one row per school — and the same two-step fix:

```python
school_counts = (
    schools.with_columns(pl.col("town").str.to_uppercase())
    .group_by("town")
    .agg(pl.col("school_name").count().alias("school_count"))
)

hdb_enriched = hdb_enriched.join(school_counts, on="town", how="left")
```

This is the general pattern: if the right-hand table is at a finer grain than the left (many schools per town, one row per transaction), normalise the key, aggregate the right-hand table up to the left table's grain, then join.

### Step 6: Fill the nulls — deliberately

Three HDB towns (BOON LAY, CENTRAL AREA, KALLANG/WHAMPOA) have no entry in the schools table, so 2,970 rows get a null `school_count`. Fill it with zero, and write down what zero means:

```python
# 0 = "no school in this table for this town", not "the town has no schools"
hdb_enriched = hdb_enriched.with_columns(pl.col("school_count").fill_null(0))
```

`.fill_null(0)` replaces every null with the given value. For counts, 0 is the usual default — but it is a modelling choice, and here it is partly wrong: Central Area certainly has schools; they are just not in this table under that name. For `station_count` we do *not* fill. Writing 0 would assert "this town has no MRT station", which is false for Hougang, Punggol and Sengkang; leaving it null says, truthfully, "we don't know". Investigate before you fill, and document every fill value.

### Step 7: Verify, and compare join types

```python
for col in ("station_count", "station_spacing_km", "school_count"):
    nc = hdb_enriched[col].null_count()
    pct = nc / hdb_enriched.height
    print(f"  {col} nulls: {nc:,} ({pct:.1%})")
```

```text
  station_count nulls: 11,032 (22.0%)
  station_spacing_km nulls: 11,032 (22.0%)
  school_count nulls: 0 (0.0%)
```

22% of sales are in the six towns the station table does not cover — exactly what Step 3 predicted. Now see what each join type would have done with them:

```python
hdb_inner = hdb.join(mrt_by_town, on="town", how="inner")
print(f"Left join:  {hdb_enriched.height:,} rows")
print(f"Inner join: {hdb_inner.height:,} rows")

town_coverage = (
    hdb.group_by("town").agg(pl.len().alias("hdb_rows"))
    .join(mrt_by_town.select("town", "station_count"), on="town", how="full", coalesce=True)
)
print(f"Towns in either table: {town_coverage.height}")
print(f"  HDB only: {town_coverage.filter(pl.col('station_count').is_null()).height}")
print(f"  MRT only: {town_coverage.filter(pl.col('hdb_rows').is_null()).height}")
```

```text
Left join:  50,150 rows
Inner join: 39,118 rows
Towns in either table: 38
  HDB only: 6
  MRT only: 11
```

The inner join silently drops 11,032 sales — every sale in the six unmatched towns, including two of the most expensive towns in the dataset. That is why left is the safe default for enrichment. The outer join (Polars calls it `how="full"`; `coalesce=True` merges the two `town` key columns into one) is the reconciliation view: 38 towns appear in at least one table, 6 only in HDB and 11 only in the station table (such as `ORCHARD` and `TUAS`, which have stations but no HDB sales in this file). An outer join is how you audit two sources against each other.

### Step 8: Build a district-level summary with spatial context

Group by town and pull through the town-level columns, then compute each town's straight-line (haversine) distance to the CBD from the centre of its stations. `math` is Python's built-in maths module — this is the third-party-free `import` mentioned at the start of the lesson:

```python
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


CBD_LAT, CBD_LNG = 1.2830, 103.8513   # Raffles Place MRT

district_summary = (
    hdb_enriched.group_by("town")
    .agg(
        pl.len().alias("total_transactions"),
        pl.col("resale_price").median().alias("median_price"),
        # Town-level columns — same for every row in a town, so .first() is correct
        pl.col("station_count").first().alias("station_count"),
        pl.col("town_lat").first().alias("town_lat"),
        pl.col("town_lng").first().alias("town_lng"),
        pl.col("school_count").first().alias("school_count"),
    )
    .sort("median_price", descending=True)
)

km_to_cbd = [
    haversine_km(lat, lng, CBD_LAT, CBD_LNG) if lat is not None else None
    for lat, lng in zip(district_summary["town_lat"], district_summary["town_lng"])
]
district_summary = district_summary.with_columns(
    pl.Series("km_to_cbd", km_to_cbd, dtype=pl.Float64)
)
print(district_summary.select("town", "median_price", "station_count", "km_to_cbd").head(5))
```

```text
shape: (5, 4)
┌─────────────────┬──────────────┬───────────────┬───────────┐
│ town            ┆ median_price ┆ station_count ┆ km_to_cbd │
│ ---             ┆ ---          ┆ ---           ┆ ---       │
│ str             ┆ f64          ┆ u32           ┆ f64       │
╞═════════════════╪══════════════╪═══════════════╪═══════════╡
│ TOA PAYOH       ┆ 973887.0     ┆ 2             ┆ 5.936218  │
│ KALLANG/WHAMPOA ┆ 964675.0     ┆ null          ┆ null      │
│ BUKIT TIMAH     ┆ 961182.0     ┆ 10            ┆ 7.717159  │
│ QUEENSTOWN      ┆ 960422.0     ┆ 7             ┆ 6.044103  │
│ CENTRAL AREA    ┆ 955722.5     ┆ null          ┆ null      │
└─────────────────┴──────────────┴───────────────┴───────────┘
```

Why `.first()` for the town-level columns? Because every transaction in a given town has the *same* value for them (they were joined from a town-level table). `.mean()` would give the same answer, but `.first()` says what you mean: "just grab one". The towns without station data have no position, so their `km_to_cbd` is `None` — the list comprehension checks for that rather than crashing.

### Step 9: Compute correlations between price and spatial features

```python
for feature in ("km_to_cbd", "station_count", "school_count"):
    r = district_summary.select(pl.corr(feature, "median_price")).item()
    print(f"Correlation: {feature:<13} <-> median price: {r:+.3f}")
```

`pl.corr(a, b)` computes the Pearson correlation between two columns, skipping towns where either value is null (so the CBD and station correlations use the 21 matched towns). `.item()` extracts the single scalar from the one-row, one-column DataFrame Polars returns.

Expected output:

```text
Correlation: km_to_cbd     <-> median price: -0.552
Correlation: station_count <-> median price: +0.256
Correlation: school_count  <-> median price: -0.189
```

Distance to the CBD has a moderate negative correlation with price: towns nearer the city centre sell for more. That is the feature that actually tracks price in this data. Station count is weakly positive, and school count is weakly *negative* — the opposite of the "good schools raise prices" story you might have expected. Three cautions before you tell anyone. First, these are 21–27 towns, so a single town can swing a correlation. Second, `school_count` includes the three towns we filled with 0, which pulls the correlation around. Third, none of this is causal: the same underlying variable — how central and desirable a town is — drives price, station density and everything else. Building a school in a town would not make it central. Correlation is information, not causation; Module 2 covers how to reason about causes properly.

## Try It Yourself

**Drill 1.** Without running the inner join, predict how many rows an inner join of `hdb` and `mrt_by_town` will produce. Use only the `unmatched` set from Step 3 and a filter on `hdb`. Then check your prediction against Step 7.

**Drill 2.** Write an `if` statement that prints `"Most towns matched"` if `len(matched) / len(hdb_towns) > 0.95`, `"Some towns missing"` if between 0.8 and 0.95, and `"Many towns missing — investigate"` otherwise. Which message does the MRT join produce?

**Drill 3.** Join `district_summary` (from Step 8) with itself using `how="cross"` (Polars' Cartesian product). Filter the result to pairs where town_a comes alphabetically before town_b (`pl.col("town") < pl.col("town_right")`). For each pair, compute the absolute difference in `median_price`. What is the largest difference, and between which towns?

**Drill 4.** Pre-aggregate the schools table to count *primary schools only* per town (the `type` column holds `"primary"`, `"secondary"` or `"JC"`). Remember the casing. Then left-join onto the HDB data and fill nulls with 0.

**Drill 5.** Use `set` operations to find towns that appear in the schools table but not in the HDB table (after upper-casing). Are there any? What does that tell you about using a town name as a join key across two independently-built tables?

## Cross-References

- **Lesson 1.5** will move into time-series analysis with window functions, which operate on the enriched joined dataset you built here.
- **Lesson 1.6** will visualise relationships between numeric columns as scatter plots and a correlation heatmap.
- **Lesson 1.7** will perform a more complex multi-source merge when aligning monthly CPI, quarterly employment, and daily FX-rate data onto a common monthly spine.
- **Module 2**'s FeatureStore uses joins under the hood to materialise feature groups. The join semantics are exactly what you learned today.

## Reflection

You should now be able to:

- Define inner, left, right, and outer joins in plain English and give an example of when each is appropriate.
- Write a Polars `.join()` call with `on`, `how`, and select-down-the-right-side.
- Predict the null-count outcome of a left join using Python `set` operations before running the join.
- Use `.fill_null()` to replace post-join NULLs with a sensible default.
- Pre-aggregate a right-hand table to the left table's grain before joining.
- Explain why `.first()` is the right aggregation for a column that was joined from a coarser-grained table.
- Compute a Pearson correlation between two columns with `df.select(pl.corr(a, b)).item()`.
- Explain why a left join can *increase* the row count, and fix it by aggregating the right-hand table to a unique key first.
- Normalise join keys (casing, naming) before joining, and use an outer (`how="full"`) join to reconcile two sources.

### Drill answers

1. Every sale in an unmatched town is dropped by an inner join, so the prediction is the HDB row count minus the rows in those towns:
   ```python
   lost = hdb.filter(pl.col("town").is_in(sorted(unmatched))).height
   print(f"Predicted inner join rows: {hdb.height - lost:,}")   # 39,118 (11,032 lost)
   ```
   This matches Step 7 exactly — because `mrt_by_town` has one row per town. Had you joined the raw station table, the inner join would also have multiplied rows, and no prediction from `unmatched` alone would hold.
2. ```python
   match_rate = len(matched) / len(hdb_towns)
   if match_rate > 0.95:
       print("Most towns matched")
   elif match_rate > 0.80:
       print("Some towns missing")
   else:
       print("Many towns missing — investigate")
   ```
   21 / 27 = 0.78, so it prints `"Many towns missing — investigate"`. (Before upper-casing, the rate was 0.)
3. ```python
   pairs = (
       district_summary.select("town", "median_price")
       .join(district_summary.select("town", "median_price"), how="cross", suffix="_right")
       .filter(pl.col("town") < pl.col("town_right"))
       .with_columns(
           (pl.col("median_price") - pl.col("median_price_right")).abs().alias("diff")
       )
       .sort("diff", descending=True)
   )
   print(pairs.head(1))
   ```
   The largest gap is BOON LAY vs TOA PAYOH: S$153,906.50, the cheapest and most expensive town medians.
4. ```python
   primary_counts = (
       schools.filter(pl.col("type") == "primary")
       .with_columns(pl.col("town").str.to_uppercase())
       .group_by("town")
       .agg(pl.len().alias("primary_school_count"))
   )
   hdb_with_primary = hdb.join(primary_counts, on="town", how="left").with_columns(
       pl.col("primary_school_count").fill_null(0)
   )
   ```
   Without `.str.to_uppercase()` nothing matches and every row gets 0 after `fill_null` — a silent, plausible-looking wrong answer.
5. ```python
   school_towns = {t.upper() for t in schools["town"].unique().to_list()}
   hdb_towns = set(hdb["town"].unique().to_list())
   only_in_schools = school_towns - hdb_towns
   print(sorted(only_in_schools))   # ['BUONA VISTA', 'KALLANG', 'LITTLE INDIA']
   ```
   Three. The schools table uses neighbourhood names (`Kallang`, `Little India`) where the HDB table uses planning-town names (`KALLANG/WHAMPOA`, `CENTRAL AREA`). Matching casing is not enough: two tables built by different people rarely agree on what a "town" is. A proper fix is a mapping table from one naming scheme to the other.

---

# Lesson 1.5: Window Functions and Trends

## Why This Matters

Aggregation answers "what is the median price per town?" but collapses every transaction into a single row per group. Sometimes you want the statistic without losing the row-level detail. "For each transaction, what was the median price in the same town in the same year?" That question cannot be answered with `group_by` alone — you would get one row per (town, year) and lose the per-transaction detail. The answer is a *window function*: a calculation that uses a group of related rows to compute a value for each row, without collapsing.

Window functions unlock time-series analysis. Rolling averages that smooth monthly noise. Year-over-year changes that reveal growth trends. Ranks within a group that identify top performers. These are all window calculations. And unlike `group_by`, window functions keep the original rows, so you can `.filter()` or `.sort()` the result just like any other DataFrame.

This lesson also introduces *lazy frames*, Polars' query-optimisation mode. You will not need lazy frames for correctness — eager evaluation works fine for everything in this chapter — but for large datasets lazy can be dramatically faster, and you should know it exists.

## Core Concepts

### FOUNDATIONS: What is a window function?

A window function computes a value for each row based on a set of surrounding rows called the *window*. The key distinction from `group_by`:

| | `group_by` | window function |
|---|---|---|
| Output rows | one per group | same as input (one per row) |
| Output columns | aggregations | new derived columns |
| Keeps row-level detail | no | yes |
| Typical use | summary tables | feature engineering, trend detection |

The Polars syntax for a window function is a column expression followed by `.over(partition_col)`:

```python
df.with_columns(
    pl.col("price").mean().over("town").alias("town_mean_price")
)
```

Reading this: "for each row, compute the mean of `price` over all rows with the same `town`, and put the result in a new column `town_mean_price`". The partition column `"town"` defines the window — for a given row, the window is every other row with the same town value.

After running, every row has its original columns plus `town_mean_price`, which is the mean price of the town that row belongs to. Two rows for Bishan get the same `town_mean_price` (they share a window). A Yishun row gets a different value (different window).

### FOUNDATIONS: Rolling averages — smoothing noisy time series

A rolling average (also called a moving average) replaces each value with the average of itself and the $k-1$ values before it, where $k$ is the window size. Rolling averages smooth out short-term noise to reveal underlying trends.

Consider a monthly median price per town. Any single month can be noisy — one unusual transaction or a slow sales month can make the median jump around. A 12-month rolling average smooths the noise: each month's smoothed value is the average of the last twelve months, so single-month anomalies are diluted by eleven other months' data.

In Polars:

```python
monthly_prices.with_columns(
    pl.col("median_price_sqm")
    .rolling_mean(window_size=12)
    .over("town")
    .alias("rolling_12m_price_sqm")
)
```

Reading this: "for each row, compute the rolling mean of `median_price_sqm` with a window of 12 rows, partitioned by town". The partition by town is critical — without `.over("town")`, the rolling window would bleed across towns, averaging Bishan prices with Yishun prices. With `.over("town")`, each town gets its own independent rolling window.

The first 11 rows in each town will be NULL, because you need 12 rows to compute a 12-row window. This is normal. When you plot the result, the line starts 11 months in.

**Choosing a window size.** The trade-off is reactivity vs smoothness:

- **Small window (3 months):** reacts quickly to price changes, still has visible noise. Useful for detecting early market turns.
- **Medium window (6 months):** middle ground.
- **Large window (12 months):** very smooth, lags by 6 months. Useful for seeing the underlying trend without noise.

A common technique borrowed from financial technical analysis is to plot two rolling averages of different sizes on the same chart. When the short moving average (say 3 months) crosses above the long one (12 months), it signals an accelerating market. When it crosses below, it signals a slowing market. This is called the "golden cross" and "death cross". It is a crude signal — more useful as a visual aid than a trading rule — but the underlying idea that short-vs-long moving averages can encode trend direction is sound.

### FOUNDATIONS: `shift` — compare to a previous row

`shift(n)` moves every value forward by $n$ positions in the DataFrame, filling the first $n$ rows with NULL. Combined with `.over("town")`, each town shifts independently.

Year-over-year change is the classic use case. "What is the median price this month compared to the same month last year?" Take the median price, shift it by 12 months, and compute the percentage difference:

```python
monthly_prices.with_columns(
    pl.col("median_price_sqm").shift(12).over("town").alias("price_sqm_12m_ago"),
).with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_12m_ago"))
        / pl.col("price_sqm_12m_ago")
        * 100
    ).alias("yoy_price_change_pct")
)
```

The first `with_columns` creates a helper column `price_sqm_12m_ago` — the value from 12 rows earlier. The second `with_columns` uses that helper to compute the percentage change. We split it into two steps for readability; you could inline it into one step, but the readability cost is not worth it.

The YoY calculation uses the standard formula:

$$\text{YoY \%} = \frac{\text{current} - \text{previous}}{\text{previous}} \times 100$$

A YoY of +5% means prices this month are 5% higher than the same month last year. A YoY of -2% means prices fell by 2%. Using the same-month comparison (shift by 12) removes the effect of monthly seasonality — comparing January to January is more meaningful than January to December, because January always has a different sales volume than December.

### FOUNDATIONS: `rank` — order within a group

Rank assigns each row a position within its partition. `pl.col("yoy").rank(method="ordinal", descending=True)` assigns 1 to the highest YoY, 2 to the next, and so on. Ranks are useful for answering "which town had the nth highest growth this year?" without sorting and inspecting manually.

The `method` parameter controls how ties are broken:

- `"ordinal"` — each value gets a distinct rank (ties broken arbitrarily).
- `"dense"` — ties get the same rank, and the next value gets the next integer. `[1, 2, 2, 3]`.
- `"min"` — ties get the lowest possible rank, and the next value skips ahead. `[1, 2, 2, 4]`.
- `"average"` — ties get the average of their ranks. `[1, 2.5, 2.5, 4]`.

Polars' own default is `"average"`; pass `method="ordinal"` when you want a strict 1-2-3 ordering. `"dense"` is useful when you want to count distinct values and assign them consecutive ranks regardless of duplicates.

### FOUNDATIONS: Trend and seasonality — and testing for them

A time series can move in two systematic ways. A **trend** is a long-run drift: prices that rise (or fall) year after year. **Seasonality** is a pattern that repeats on a fixed calendar cycle: more sales every March, higher prices every December. Everything else is noise.

Window functions are your tools for both. A long rolling mean (12 months) averages away seasonality and noise, leaving the trend. A YoY change compares each month to the same month a year earlier, so a seasonal pattern cancels out and only trend plus noise remain. To look for seasonality directly, group by *month of the year* (1–12) across all years and compare: if every January is high, the January group will stand out.

The important habit is to treat "there is a seasonal pattern" (or "prices are rising") as a **hypothesis to test**, not a fact to assume. Housing markets in many countries do have seasonal rhythms and long-run trends. Whether *this* dataset has them is a question only the data can answer — and in the worked example below, the honest answer turns out to be no.

### FOUNDATIONS: Lazy frames — query optimisation

Every Polars operation so far has been *eager*: you wrote an expression, Polars computed it immediately, and the result was a DataFrame. Eager evaluation is simple and what you want for exploration.

For large data and complex pipelines, Polars offers a *lazy* mode. A lazy DataFrame (`LazyFrame`) is a description of a query, not the query's result. You chain operations as normal, and each operation adds to the query plan without running anything. When you finally call `.collect()`, Polars optimises the entire plan and then executes it.

```python
result = (
    monthly_prices.lazy()
    .filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
    .drop_nulls("yoy_price_change_pct")
    .group_by("town")
    .agg(pl.col("yoy_price_change_pct").mean().alias("mean_yoy_pct"))
    .sort("mean_yoy_pct", descending=True)
    .collect()
)
```

The `.lazy()` call converts the eager `monthly_prices` into a LazyFrame. Everything between `.lazy()` and `.collect()` is query-plan construction, not execution. The `.collect()` at the end triggers the whole pipeline at once, with optimisations.

The optimisations include:

- **Predicate pushdown.** Filters are pushed down the query plan so that rows are eliminated as early as possible, before subsequent operations have to touch them. If you filter to 2021+ *after* a group_by, the optimiser can usually move the filter *before* the group_by, so the group_by processes fewer rows.
- **Projection pushdown.** Unused columns are dropped as early as possible. If the final output uses only two columns out of twenty, the optimiser removes the other eighteen before they reach the heavy operations.
- **Operation fusion.** Adjacent operations that can be combined into a single pass over the data are fused. Multiple `with_columns` calls become one. Multiple filters become one.

For small datasets (thousands of rows), lazy and eager take the same time. For large datasets (millions or more), lazy can be 2–10× faster. Always profile before you switch to lazy — premature lazy adds complexity without benefit.

You can also read files lazily with `pl.scan_csv` or `pl.scan_parquet`. These return a LazyFrame without reading the file; the file is only read when you `.collect()`, and only the columns you actually need are read from disk. For a wide CSV with a hundred columns where your query uses only three, scan can be 30× faster than read.

### THEORY: When window functions fail

Window functions assume the ordering is correct. If you compute a rolling mean over a DataFrame that is not sorted by date, the "previous 12 rows" are whatever happens to come before in the current row order, which might not be the previous 12 months. Always sort before computing a window function:

```python
monthly_prices = monthly_prices.sort("town", "transaction_date")
```

Sort by partition key first (so rows in the same partition are adjacent), then by the ordering key within each partition. After sorting, `.over("town")` correctly partitions and the window functions compute what you expect.

There is also a subtlety with `shift` and `rolling_mean`: in a window context they count *rows*, not time. If a month is missing from the series (a month with no transactions in that town), `shift(12)` does *not* reach back exactly 12 calendar months — it reaches back 12 rows, which is 13 calendar months for every row after the gap. Nothing warns you; the YoY number is simply wrong. The course HDB data has exactly this problem: 27 towns × 120 months = 3,240 town-months, but only 3,236 have any sales (Bukit Timah is missing three months, Central Area one). A naive `shift(12).over("town")` silently misaligns 40 rows.

The fix is to build a complete **calendar spine** — every town × every month — and left-join the observed data onto it before any window function. Missing months then become explicit null rows, so a 12-row shift is always a 12-month shift, and a rolling window containing a gap returns null rather than quietly averaging across it. The worked example does exactly this. (The alternative is a join on a date-shifted copy of the table, e.g. matching each row to `transaction_date.dt.offset_by("-12mo")`.)

### ADVANCED: Rolling windows are a form of convolution

A rolling mean is equivalent to convolving the time series with a box filter (a kernel of all ones divided by the window size). This is why rolling means smooth data: they are low-pass filters that remove high-frequency noise. More sophisticated window functions (weighted rolling means, exponentially weighted moving averages) are just different convolution kernels.

If you want to emphasise recent data over old data, an exponentially weighted moving average gives higher weight to recent observations. Polars supports this with `.ewm_mean(alpha=...)`. The `alpha` parameter controls decay: higher `alpha` means faster forgetting of old data.

This is the discrete-time analogue of the IIR filters used in signal processing. The connection matters when you reach Module 5 and learn about temporal convolutional networks, which are exactly the same idea applied to raw neural network features instead of statistical features.

## The Kailash Context

The time-series analysis you are learning here feeds directly into `FeatureEngineer` in Module 2 and `TrainingPipeline` in Module 3. Rolling means, YoY changes, and ranks within group are common features in any production ML pipeline — they are how models learn "is the price higher than it was last year?" rather than "is the price high?". The Kailash `FeatureEngineer` engine has built-in methods for generating lag features, rolling statistics, and rank features automatically, but the underlying computation is what you just learned in pure Polars. Knowing the manual form is what lets you debug when the automated version produces unexpected results.

## Worked Example: HDB Price Trends by Town

### Step 1: Prepare the time-series base table

```python
from __future__ import annotations

from datetime import date

import polars as pl

from shared import MLFPDataLoader

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
)

monthly_observed = (
    hdb.group_by("town", "transaction_date")
    .agg(
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.col("resale_price").median().alias("median_resale_price"),
        pl.len().alias("transaction_count"),
    )
)
print(f"Town-month rows observed: {monthly_observed.height:,}")

# Calendar spine: every town x every month, so a 12-row shift is a 12-month shift
towns = hdb.select("town").unique()
months = pl.DataFrame({
    "transaction_date": pl.date_range(date(2015, 1, 1), date(2024, 12, 1), "1mo", eager=True)
})
spine = towns.join(months, how="cross")
print(f"Complete town x month grid: {spine.height:,}")

monthly_prices = (
    spine.join(monthly_observed, on=["town", "transaction_date"], how="left")
    .sort("town", "transaction_date")
)
print(monthly_prices.filter(pl.col("transaction_count").is_null()).select("town", "transaction_date"))
```

Expected output:

```text
Town-month rows observed: 3,236
Complete town x month grid: 3,240
shape: (4, 2)
┌──────────────┬──────────────────┐
│ town         ┆ transaction_date │
│ ---          ┆ ---              │
│ str          ┆ date             │
╞══════════════╪══════════════════╡
│ BUKIT TIMAH  ┆ 2015-10-01       │
│ BUKIT TIMAH  ┆ 2017-02-01       │
│ BUKIT TIMAH  ┆ 2022-07-01       │
│ CENTRAL AREA ┆ 2015-08-01       │
└──────────────┴──────────────────┘
```

`pl.date_range(..., "1mo", eager=True)` builds the 120 month-start dates from January 2015 to December 2024, and the cross join pairs each with each of the 27 towns. Left-joining the observed medians onto that grid turns the four missing town-months into explicit null rows. The base table now has exactly one row per (town, month) with no gaps, sorted by town then date — the sort is crucial, because window functions read rows in their current order.

### Step 2: Rolling averages

```python
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").rolling_mean(window_size=12).over("town").alias("rolling_12m_price_sqm"),
    pl.col("median_price_sqm").rolling_mean(window_size=3).over("town").alias("rolling_3m_price_sqm"),
)

print(monthly_prices.filter(pl.col("town") == "BISHAN").tail(18))
```

Inspect Bishan for the last 18 months. With only 11–18 sales in a month, the raw `median_price_sqm` bounces by several hundred dollars from month to month (S$9,262 to S$10,159 in the second half of 2024). `rolling_3m_price_sqm` follows it more smoothly, and `rolling_12m_price_sqm` barely moves (about S$9,890–10,080). When `rolling_3m` rises above `rolling_12m`, the short-term average is above the long-term average — but in a flat series like this, those crossings are noise, not market signals. A rolling window that contains one of the spine's null months returns null, so Bukit Timah's 12-month average disappears for a year after each gap instead of silently spanning it.

### Step 3: Year-over-year change

```python
monthly_prices = monthly_prices.with_columns(
    pl.col("median_price_sqm").shift(12).over("town").alias("price_sqm_12m_ago"),
)

monthly_prices = monthly_prices.with_columns(
    (
        (pl.col("median_price_sqm") - pl.col("price_sqm_12m_ago"))
        / pl.col("price_sqm_12m_ago")
        * 100
    ).alias("yoy_price_change_pct")
)

print(monthly_prices.filter(pl.col("town") == "BISHAN").tail(24).select(
    "transaction_date", "median_price_sqm", "price_sqm_12m_ago", "yoy_price_change_pct"
))
```

In the last six months of 2024, Bishan's YoY values run from about −10% to +0.2%. Look across the whole table and the picture is clear: over all towns and months the YoY change averages +0.26% with a standard deviation of about 6 percentage points (range −29% to +32%). Individual months swing wildly because each town-month median is built from a dozen or so sales, but there is no sustained growth. This is the synthetic dataset's flat price level showing through — the real resale market rose strongly after 2020, and a learner who expects that will "see" a rebound in noise if they do not check.

### Step 4: Find the trend leaders with lazy evaluation

This is where lazy frames pay for themselves. The multi-step pipeline below (filter, drop nulls, group_by, sort) has enough operations that the optimiser can meaningfully rewrite it:

```python
recent_yoy = (
    monthly_prices.lazy()
    .filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
    .drop_nulls("yoy_price_change_pct")
    .group_by("town")
    .agg(
        pl.col("yoy_price_change_pct").mean().alias("mean_yoy_pct"),
        pl.col("yoy_price_change_pct").std().alias("std_yoy_pct"),
        pl.col("yoy_price_change_pct").max().alias("peak_yoy_pct"),
        pl.col("yoy_price_change_pct").min().alias("trough_yoy_pct"),
        pl.len().alias("months_of_data"),
    )
    .sort("mean_yoy_pct", descending=True)
    .collect()
)

print(recent_yoy.head(10))
```

Each town's `mean_yoy_pct` is its average YoY change since 2021. In this data the "leader", BUKIT TIMAH, averages +1.35% with a standard deviation of 10 points and a peak of +28%, and the bottom town, SERANGOON, averages −0.62%. Every town's mean is within about ±1.4% of zero while its monthly swings are ten times larger. Compare `mean_yoy_pct` to `std_yoy_pct` before you call a town a leader: a mean that is small relative to its own spread is indistinguishable from zero. Note also `months_of_data`: Bukit Timah has 46 rather than 48 because its 2022-07 gap removes two YoY values (2022-07 itself and 2023-07, whose 12-month-ago value is the gap).

### Step 5: Classify towns into leaders, followers, and laggards

Use the mean and standard deviation of the growth rates to segment:

```python
mean_growth = recent_yoy["mean_yoy_pct"].mean()
std_growth = recent_yoy["mean_yoy_pct"].std()

recent_yoy = recent_yoy.with_columns(
    pl.when(pl.col("mean_yoy_pct") > mean_growth + std_growth).then(pl.lit("leader"))
    .when(pl.col("mean_yoy_pct") < mean_growth - std_growth).then(pl.lit("laggard"))
    .otherwise(pl.lit("follower"))
    .alias("trend_category"),
    pl.col("mean_yoy_pct").rank(method="ordinal", descending=True).alias("growth_rank"),
)

print(f"Mean YoY growth (all towns): {mean_growth:.2f}%")
print(f"Std dev: {std_growth:.2f}%")
print(recent_yoy.group_by("trend_category").agg(pl.len().alias("count")))
```

Expected output:

```text
Mean YoY growth (all towns): 0.20%
Std dev: 0.39%
shape: (3, 2)
┌────────────────┬───────┐
│ trend_category ┆ count │
│ ---            ┆ ---   │
│ str            ┆ u32   │
╞════════════════╪═══════╡
│ follower       ┆ 22    │
│ laggard        ┆ 3     │
│ leader         ┆ 2     │
└────────────────┴───────┘
```

(The order of `group_by` output rows is not guaranteed; add `.sort("trend_category")` to fix it.) Under a roughly normal distribution, about 16% of towns should be more than one standard deviation above the mean and 16% below — about 4 of 27 each; here there are 2 leaders and 3 laggards. But notice the scale: the segmentation is relative, so it *always* produces leaders and laggards, even when the whole spread is 0.39 percentage points. A classification rule built on mean ± std tells you who is above the others, never whether the difference matters.

### Step 6: Test for seasonality

Is there a best month of the year to buy? Group every sale by its calendar month, across all ten years:

```python
seasonal = (
    hdb.with_columns(pl.col("transaction_date").dt.month().alias("month_of_year"))
    .group_by("month_of_year")
    .agg(
        pl.len().alias("transactions"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
    )
    .sort("month_of_year")
)
print(seasonal)
```

The 12 rows show between 4,001 (January) and 4,333 (November) sales per calendar month, and median prices per square metre between S$8,759 (August) and S$8,836 (October) — a spread of S$78, under 1%. There is no month that is consistently busier or dearer. The hypothesis "prices are seasonal" is rejected for this dataset, just as the "prices are trending up" hypothesis was rejected in Step 3. Both are correct, useful findings: you checked rather than assumed, and you now know that any model trained on this file will find no calendar signal to learn.

## Try It Yourself

**Drill 1.** Add a `rolling_6m_price_sqm` column using a 6-month window. Plot or print the last 24 months for Queenstown along with the raw, 3m, 6m, and 12m rolling values.

**Drill 2.** Compute month-over-month (MoM) percentage change — shift by 1 instead of 12. For Bishan, print the 12 months with the largest absolute MoM changes. Are they clustered in time?

**Drill 3.** Rank towns by `peak_yoy_pct` (their single best month of growth) and print the top 5 along with the month it occurred. (Hint: first find the row where peak_yoy_pct was reached per town.)

**Drill 4.** Rewrite Step 4 in eager mode (no `.lazy()` / `.collect()`). Confirm the output is identical. Time both versions using Python's `time.perf_counter()`; which is faster?

**Drill 5.** Compute a *compound* annual growth rate (CAGR) per town from January 2021 to December 2024, using the `rolling_12m_price_sqm` values at the two ends (smoother than single months). CAGR = (ending / beginning)^(1/years) − 1, where `years` must be the actual time between the two dates. Which town has the highest CAGR?

## Cross-References

- **Lesson 1.6** will visualise the rolling averages and YoY trends as line charts — the natural chart type for time-series data.
- **Lesson 1.7** will align monthly, quarterly and daily economic series onto one calendar — the same spine idea you used here.
- **Module 2**'s `FeatureEngineer` automates rolling-feature generation at scale.
- **Module 5** will reintroduce rolling windows as a form of convolution when you meet TCNs (temporal convolutional networks).

## Reflection

You should now be able to:

- Explain the difference between `group_by` and window functions, and decide which to use for a given question.
- Write a Polars `.rolling_mean()` over a partition column with a specified window size.
- Write a `.shift(n).over(col)` to compute a value from $n$ rows earlier within each partition.
- Compute YoY percentage change using shift and arithmetic.
- Explain what lazy evaluation means, what `.collect()` does, and what optimisations the Polars query planner performs.
- Use `rank` with different tie-breaking methods and understand what each produces.
- Classify values into categories using `pl.when().then()` combined with the mean and standard deviation.
- Build a complete calendar spine before using `shift` or `rolling_mean`, and explain why gaps break row-based windows.
- Test a trend or seasonality hypothesis with YoY changes and a month-of-year group_by, and report a negative result honestly.

### Drill answers

1. ```python
   monthly_prices = monthly_prices.with_columns(
       pl.col("median_price_sqm").rolling_mean(window_size=6).over("town").alias("rolling_6m_price_sqm")
   )
   print(monthly_prices.filter(pl.col("town") == "QUEENSTOWN").tail(24))
   ```
2. ```python
   monthly_prices = monthly_prices.with_columns(
       pl.col("median_price_sqm").shift(1).over("town").alias("price_sqm_1m_ago"),
   )
   monthly_prices = monthly_prices.with_columns(
       ((pl.col("median_price_sqm") - pl.col("price_sqm_1m_ago")) / pl.col("price_sqm_1m_ago") * 100).alias("mom_pct")
   )
   top = (
       monthly_prices.filter(pl.col("town") == "BISHAN")
       .drop_nulls("mom_pct")
       .with_columns(pl.col("mom_pct").abs().alias("abs_mom"))
       .sort("abs_mom", descending=True)
       .head(12)
   )
   print(top)
   ```
3. ```python
   peaks = (
       monthly_prices.drop_nulls("yoy_price_change_pct")
       .sort("yoy_price_change_pct", descending=True)
       .group_by("town")
       .agg(
           pl.col("yoy_price_change_pct").first().alias("peak_yoy"),
           pl.col("transaction_date").first().alias("peak_month"),
       )
       .sort("peak_yoy", descending=True)
       .head(5)
   )
   print(peaks)
   ```
4. Remove `.lazy()` and `.collect()`; the output is identical. On a 3,240-row table both versions take a few milliseconds and the timing difference is noise — sometimes one wins, sometimes the other. Lazy evaluation pays off on large inputs (millions of rows, or `pl.scan_parquet` reading only the needed columns from a big file), not here.
5. ```python
   cagr = (
       monthly_prices.filter(pl.col("transaction_date") >= pl.date(2021, 1, 1))
       .drop_nulls("rolling_12m_price_sqm")
       .group_by("town")
       .agg(
           pl.col("rolling_12m_price_sqm").first().alias("start"),
           pl.col("rolling_12m_price_sqm").last().alias("end"),
           pl.col("transaction_date").first().alias("first_date"),
           pl.col("transaction_date").last().alias("last_date"),
       )
       .with_columns(
           ((pl.col("last_date") - pl.col("first_date")).dt.total_days() / 365.25).alias("years")
       )
       .with_columns(
           ((pl.col("end") / pl.col("start")) ** (1 / pl.col("years")) - 1).alias("cagr")
       )
       .sort("cagr", descending=True)
   )
   print(cagr.head(3))
   ```
   The exponent uses the real elapsed time (about 3.9 years from 2021-01 to 2024-12), not a hard-coded `1/3`. The highest is BUKIT TIMAH at about 0.86% a year, and every town lies between about −0.8% and +0.9% — flat, as Step 3 found.

---

# Lesson 1.6: Data Visualisation

## Why This Matters

Anscombe's quartet is a set of four datasets, each with eleven points. Every dataset has the same mean of x, the same mean of y, the same variance of x, the same variance of y, the same correlation, and the same regression line. By the numbers they are indistinguishable. By the *pictures* they are completely different: one is a clean linear relationship, one is a smooth curve, one is a line with a single outlier that dominates the correlation, and one is a vertical line of points plus a single outlier at the far right. The lesson from Anscombe's quartet is simple and severe: *never trust a summary statistic you have not plotted.*

Every descriptive statistic you learned in Lessons 1.1 through 1.5 is a compression. Compressions lose information. A mean of $500,000 tells you nothing about whether the underlying distribution is a tight bell or a bimodal mess. A correlation of 0.6 could mean a clean linear trend or a parabola with a single outlier. The only way to see what the data is actually doing is to plot it.

This lesson is about choosing the right chart for the right question and building it without introducing distortions. You will learn six chart types — histogram, scatter, bar, heatmap, line, stacked bar — each with a specific job. You will learn the Gestalt principles that govern what the eye can parse quickly and what it cannot. You will learn which chart types to avoid (3D charts, pie charts in most circumstances) and why. And you will build them with `ModelVisualizer`, the Kailash engine that wraps Plotly behind a consistent API, dropping down to Plotly itself for the two chart types ModelVisualizer does not provide (heatmap and stacked bar).

## Core Concepts

### FOUNDATIONS: Why visualise?

Tables are precise. Charts are fast. A well-chosen chart communicates a pattern in milliseconds that would take a minute to extract from a table. The human visual system is the most parallel sensor on your body — you can spot an outlier in a scatter of ten thousand points instantly, whereas extracting that same outlier from a table would require scanning ten thousand rows.

The trade-off is precision. A chart cannot tell you that the exact maximum value was $1,248,532.50; a table can. So charts and tables are complementary, not substitutes. Use charts to find the patterns; use tables to nail down the numbers once you know which ones matter.

The rule: **plot first, compute later.** When you are exploring a new dataset, the first thing you should do after loading it is plot the distribution of every numeric column. Only then compute statistics. This prevents the entire category of error where you report a mean of a bimodal distribution as though it were meaningful.

### FOUNDATIONS: Chart selection by data question

Different questions need different chart types. A rough mapping:

| Question | Chart type |
|---|---|
| What is the distribution of X? | histogram or density plot |
| How does Y relate to X? | scatter plot |
| How does Y differ across categories? | bar chart |
| How does Y change over time? | line chart |
| Are pairs of variables correlated? | heatmap |
| How does the distribution of Y vary across groups? | box plot or violin plot |
| What is the composition of X? (parts of a whole) | stacked bar, pie (with warnings) |

If your question does not fit one of these cleanly, chances are the question needs to be broken into parts. "How do HDB prices vary by town over time?" is two questions — across towns (bar) and over time (line) — and you will usually want two charts, or a single line chart with one line per town.

### FOUNDATIONS: Histograms and distributions

A histogram divides a continuous range of values into buckets (*bins*) and counts how many values fall into each bucket. The result is a bar chart where the x-axis is the value and the y-axis is the count. Histograms show the *shape* of a distribution:

- A **symmetric bell** — most values near the centre, tapering equally on both sides. Temperature, height, measurement error. Mean ≈ median.
- A **right-skewed** (long right tail) — most values are low with a few very high ones. Income, house prices, city populations. Mean > median.
- A **left-skewed** (long left tail) — most values are high with a few very low ones. Age at death for a mortality dataset, test scores near a ceiling. Mean < median.
- A **bimodal** — two distinct humps. The presence of two subpopulations mixed together — for example, genuine sales mixed with a batch of mis-recorded ones.
- A **uniform** — all values equally likely. Dice rolls, randomly sampled timestamps.

The *number of bins* matters. Too few bins obliterates detail — a 5-bin histogram turns everything into a rough pyramid. Too many bins makes the chart noisy — a 500-bin histogram on 1,000 data points looks like random static. Start with 30–50 bins for a few thousand data points and adjust from there. There is no universally correct rule; inspect and iterate.

### FOUNDATIONS: Scatter plots and relationships

A scatter plot puts one variable on the x-axis and another on the y-axis, with one dot per observation. It is the standard tool for asking "does X predict Y?"

What to look for:

- **Linear trend.** A cloud of points that roughly follows a straight line indicates a linear relationship. The tightness of the cloud around the line tells you how strong the relationship is.
- **Non-linear trend.** A cloud that follows a curve (parabola, exponential, logarithmic) indicates a non-linear relationship. Pearson correlation will understate these — the correlation could be near zero even though there is a strong relationship.
- **Heteroscedasticity.** If the cloud fans out (gets wider) as X increases, the relationship has non-constant variance. This matters for linear regression assumptions; we cover it in Module 2.
- **Outliers.** Points far from the main cloud are outliers. Sometimes they are real rare events; sometimes they are data errors. A scatter plot makes them instantly visible in a way no summary statistic does.
- **Clusters.** Two or three dense regions of points with sparse areas between them suggest subpopulations. This is the scatter-plot equivalent of a bimodal histogram.

For datasets with more than a few thousand points, raw scatter plots become unreadable (the dots overlap into a blob). Solutions: *sample* the data to a few thousand points, *use transparency* (alpha blending) so dense regions appear darker, or *bin the scatter* into a 2D histogram (also called a *hexbin plot*).

### FOUNDATIONS: Bar charts for categorical comparison

A bar chart has one bar per category with the bar height (or length) proportional to a value. Bar charts are the right tool for "show me this metric for each category" — median price per town, count per flat type, revenue per product.

**Vertical vs horizontal.** Vertical bar charts (categories on x-axis, values on y-axis) are the default. But when the category labels are long (like "BUKIT BATOK EAST AVENUE"), vertical bars force the labels to tilt or wrap, and the eye has to work harder. Horizontal bar charts with the labels on the y-axis and the bars extending to the right make long labels trivially readable. Use horizontal bars when labels are long or when there are many categories.

**Always sort.** An unsorted bar chart makes the reader work to find the maximum. A chart sorted descending by value shows the ranking immediately. The only exception is when the x-axis has a natural order (months of the year, age groups) that sorting by value would break.

**Start bars at zero.** A bar chart with a truncated y-axis (starting at, say, $400,000 instead of $0) visually exaggerates small differences. Three bars of heights 420, 440, and 460 look like massive differences on a truncated axis and trivial ones on a zero-based axis. For *comparison* purposes, the zero baseline is part of the chart's honesty. Exception: when the differences are meaningful but small compared to the absolute value (say, you are comparing 99.8% vs 99.9% accuracy), a truncated axis is appropriate — but document it explicitly.

### FOUNDATIONS: Line charts for time series

A line chart connects dots with line segments in the x-axis order. The implicit assumption is that the x-axis has a natural ordering — typically time. Line charts are the right tool for "how does this value change over time?"

What to look for:

- **Trend.** An overall upward or downward direction over the full range.
- **Seasonality.** A repeating pattern with a fixed period (weekly, monthly, yearly) — for example, retail sales peaking every December. Test for it rather than assume it: Lesson 1.5 found none in the course HDB data.
- **Level shifts.** Sudden jumps where the series moves to a new baseline and stays there. Indicates a regime change: a policy update, a recession, a product launch.
- **Outliers.** Spikes that return to baseline. Indicates a one-off event.

Multiple lines on one chart work well when you are comparing a few series — typically up to 5–7 lines. Beyond that the chart becomes a "spaghetti plot" and no single line is readable. For many series, use *small multiples* instead — a grid of small charts, one per series, all with the same axes.

### FOUNDATIONS: Heatmaps for correlation

A heatmap is a grid of coloured cells where colour encodes a numeric value. The two main uses are correlation matrices (showing Pearson correlation between every pair of variables in a dataset) and confusion matrices (showing classification errors — you will meet these in Module 3).

For correlation matrices, use a *diverging* colour scale: one colour for negative, another for positive, white in the middle at zero. `RdBu_r` (red-blue reversed) is a standard choice — red for positive, blue for negative, white for zero. (Plain `RdBu` runs the other way; the `_r` suffix reverses it so that "hot" red means a strong positive relationship.) This makes the sign of the correlation pre-attentively visible; you do not need to read the numbers to see which pairs are positively or negatively correlated.

Always cap the colour scale at -1 and +1 (`zmin=-1, zmax=1`). Without capping, the colour scale would stretch to fit the data, which makes different heatmaps incomparable and can wash out the meaning.

The diagonal of a correlation matrix is always 1 (every variable correlates perfectly with itself). The matrix is symmetric across the diagonal. You can show only the upper or lower triangle to reduce redundancy, but most tools do not do this by default.

### FOUNDATIONS: Gestalt principles

The Gestalt principles are a set of rules about how the human visual system groups elements. They come from early 20th century psychology but apply directly to chart design. The six that matter most for you:

- **Proximity.** Elements close to each other are perceived as belonging to the same group. Use small gaps between related bars, larger gaps between unrelated ones.
- **Similarity.** Elements that share a visual property (colour, shape, size) are perceived as belonging together. Use the same colour for the same series across multiple charts. Do not use colour randomly.
- **Closure.** The visual system fills in gaps to see complete shapes. A line chart with a missing segment is still interpreted as a continuous line. But if gaps are meaningful (missing data), make them visually distinct.
- **Continuity.** Smooth continuous lines draw the eye across a chart. This is why line charts work: the continuous line guides you through the temporal progression.
- **Connection.** Elements connected by a line are perceived as strongly related. A scatter plot with a fitted line emphasises the relationship more than the cloud alone.
- **Enclosure.** Elements inside a shared boundary — a shaded band, a box, a panel background — are perceived as a group, even if they are far apart. Shade the COVID months on a time series, or draw a light box around the three towns you are discussing, and the reader groups them instantly. Enclosure is one of the strongest grouping cues, so use it sparingly and only for the group you want noticed.

The practical rule: make the elements you want the reader to compare look *similar* to each other (same colour, same style), and make the elements you want them to distinguish look *different*. Every deviation from that rule is a potential source of confusion.

### FOUNDATIONS: Charts to avoid

**3D charts.** 3D bar charts, 3D pie charts, 3D scatter plots — all of them. The third dimension adds no information and introduces perspective distortion: bars in the back look smaller than bars in the front even if they represent the same value. The reader has to mentally correct for the perspective, which defeats the purpose of a chart. Exception: genuinely three-dimensional data (a fitted surface over two input variables, a 3D point cloud for spatial analysis). Even then, 2D projections are usually clearer.

**Pie charts, almost always.** Pie charts require the reader to compare angles, which the human visual system does poorly compared to comparing lengths. A horizontal bar chart communicates the same information (parts of a whole) more precisely. Pie charts are tolerable only when there are 2–3 slices and the approximate proportions are more important than the exact values. For anything else, use a bar chart.

**Dual y-axis line charts.** A line chart with two lines on two different y-axes — one axis on the left, another on the right — implies a relationship between the two series that may not exist. The reader's eye sees the lines crossing or diverging and infers meaning from the visual relationship, but the crossing is an artifact of how the axes were scaled. Better: use two separate charts stacked vertically, with aligned x-axes.

**Truncated y-axes on bar charts.** Covered above. A bar chart should start at zero unless you explicitly label the axis as truncated.

### FOUNDATIONS: Z-pattern reading

Western readers scan visual content in a Z pattern — top-left, across to top-right, down-left to bottom-left, across to bottom-right. Chart layout should respect this. Put the most important information in the top-left (title, key takeaway). Put secondary information in the top-right and bottom-left. Put least important information in the bottom-right. For dashboards, arrange charts so the "headline" chart is top-left and the supporting detail flows along the Z.

This is more about dashboard layout than single-chart design, but when you build reports in Lesson 1.8 you will apply it.

## The Kailash Engine: ModelVisualizer

`ModelVisualizer` is the Kailash ML engine for producing charts. It wraps Plotly under a consistent API so that every chart type uses the same calling convention. You do not have to remember that Plotly's histogram is `px.histogram` while its scatter is `px.scatter` and its bar is `go.Bar`; you just call methods on a `ModelVisualizer` instance:

```python
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
fig = viz.histogram(data=hdb, column="resale_price", bins=40, title="...")
fig.write_html("histogram.html")
```

Every ModelVisualizer method returns a Plotly `Figure` object. You can:

- **Display inline in Jupyter:** `fig.show()` renders the chart in the notebook.
- **Export to HTML:** `fig.write_html("name.html")` saves a standalone HTML file with the chart embedded. The HTML file is interactive — hover, zoom, pan all work — and can be emailed or posted.
- **Customise further:** `fig.update_layout(...)` lets you tweak titles, axes, colours, margins, and anything else Plotly supports. The underlying object is a real Plotly figure; ModelVisualizer just made the initial construction easy.

The ModelVisualizer methods you will use in Module 1, and the two chart types you build with Plotly directly:

| Method | Chart type | Typical use | Watch out for |
|---|---|---|---|
| `viz.histogram(data, column, bins=…, title=…)` | histogram | distribution of a numeric column | |
| `viz.scatter(data, x, y, color=…, title=…)` | scatter plot | relationship between two numeric columns | |
| `viz.box_plot(data, column, group_by=…, title=…)` | box plot | distribution per group | |
| `viz.metric_comparison({name: {metric: value}})` | vertical grouped bars | comparison across categories | default axis titles are "Model"/"Score" — set real ones |
| `viz.training_history({series: [values]}, x_label=…, y_label=…)` | line chart | time series | plots against 1..N — set the real x-values |
| `px.imshow(matrix, …)` (Plotly) | heatmap | correlation matrix | |
| `px.bar(df, x=…, y=…, color=…)` (Plotly) | stacked bar | composition | |

Some method names hint at their origins in ML pipelines (`training_history` was designed for plotting training loss curves, `metric_comparison` for comparing model scores). They can be repurposed for general charts, but a repurposed chart keeps its original labels — "Epoch" on the x-axis, 1..N instead of years, "Model"/"Score" — until you fix them, and a mislabelled axis is a misleading chart. Two methods cannot be repurposed at all: `confusion_matrix(y_true, y_pred, labels)` builds a classification confusion matrix from label vectors (it cannot draw an arbitrary grid such as a correlation matrix), and `feature_importance` needs a fitted model. You will use both for their real purpose in Module 3.

Polars DataFrames work directly as input — both to ModelVisualizer and to Plotly Express. You do not need to convert to pandas. This is the "polars-native" principle at work.

## Worked Example: Six HDB Charts

### Step 0: Imports and data prep

```python
from __future__ import annotations

import plotly.express as px
import polars as pl
from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

viz = ModelVisualizer()
```

(`ModelVisualizer()` prints an `ExperimentalWarning` from kailash-ml when it is created. It is informational — the engine works — and you can ignore it.)

### Step 1: Histogram of resale prices — and what it exposes

```python
fig_hist = viz.histogram(
    data=hdb,
    column="resale_price",
    bins=40,
    title="HDB Resale Price Distribution (raw)",
)
fig_hist.write_html("hdb_price_histogram_raw.html")
```

Open the HTML file in a browser. You will *not* see a nice bell or a skewed hump. With 40 bins stretched from S$10 to S$9,000,000, each bin is about S$225,000 wide: almost every sale is squashed into a few bars on the left, and a lone bar sits at the far right — the 144 planted S$9,000,000 sales. That is the histogram doing its most important job. It took one line to expose what Lesson 1.2's `.describe()` hinted at (max 9,000,000, skewness 11.4): the column contains impossible values, and they dominate the axis.

To see the real shape, set the planted values aside. The cut-offs below are a provisional choice for charting — Lesson 1.7 profiles the data properly and Lesson 1.8 cleans it:

```python
hdb_clean = hdb.filter(pl.col("resale_price").is_between(50_000, 5_000_000))
print(f"{hdb.height - hdb_clean.height} rows set aside")   # 251 rows set aside

fig_hist = viz.histogram(
    data=hdb_clean,
    column="resale_price",
    bins=40,
    title="HDB Resale Price Distribution (planted errors removed)",
)
fig_hist.write_html("hdb_price_histogram.html")
```

Now the shape is visible: a single broad hump centred near the median of about S$850k, running from about S$215k to S$1.8M, with 90% of sales between about S$474k and S$1.34M. It is only mildly right-skewed (skewness 0.39, down from 11.4), so mean and median are close. The lesson generalises: always plot the raw distribution first, because the chart of the *raw* data is the one that tells you whether the data can be trusted.

### Step 2: Scatter plot of price vs floor area

A scatter of 50,000 points is hard to read — the dots overlap into a solid blob. Sample first:

```python
hdb_sample = hdb_clean.sample(n=5_000, seed=42)

fig_scatter = viz.scatter(
    data=hdb_sample,
    x="floor_area_sqm",
    y="resale_price",
    title="HDB Resale Price vs Floor Area",
)
fig_scatter.write_html("hdb_scatter.html")
```

`hdb_clean.sample(n=5_000, seed=42)` picks 5,000 rows at random. The `seed=42` makes the sample reproducible — running the code twice with the same seed produces the same sample. Reproducibility is a lifesaver when you are debugging and want to be sure a result you saw earlier is not an artifact of random sampling.

The result is a tight, rising band: larger flats cost more. Measure it: on the cleaned data, the correlation between floor area and price is r = 0.91, so area alone accounts for about 83% of the price variance ($r^2$). Now run the same correlation on the raw `hdb`: r = 0.47, and $r^2$ falls to about 22%. The relationship did not change; 251 bad rows (0.5% of the data) hid it. Had you sampled from the raw frame, about 25 of your 5,000 dots would sit at S$10 or S$9M, stretching the y-axis until the real band became a flat smear.

### Step 3: Bar chart of median price by town

First aggregate, then plot:

```python
district_prices = (
    hdb_clean.group_by("town")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.len().alias("transaction_count"),
    )
    .sort("median_price", descending=True)
)

price_by_town = {
    town: {"Median Price (S$)": price}
    for town, price in zip(
        district_prices["town"].to_list(),
        district_prices["median_price"].to_list(),
    )
}

fig_bar = viz.metric_comparison(price_by_town)
# metric_comparison was built for model scores: its default axis titles are "Model"/"Score"
fig_bar.update_layout(
    title="Median HDB Price by Town",
    xaxis_title="Town",
    yaxis_title="Median resale price (S$)",
)
fig_bar.write_html("hdb_bar.html")
```

The `metric_comparison` method was designed for comparing metrics across models — it takes a dict of `{model_name: {metric_name: value}}` and draws vertical grouped bars, one group per key, in the order of the dict. We repurpose it by treating each town as a "model". Because we sorted `district_prices` before building the dict, the bars come out ranked. Its default axis titles are "Model" and "Score", which would mislabel the chart — so we set the real titles. When you repurpose a tool, always check its labels. For 27 long town names, a horizontal bar reads better still: `px.bar(district_prices, x="median_price", y="town", orientation="h")`.

The chart shows the step you found in Lesson 1.3: seven central towns well above the rest, then twenty towns at almost the same level. Keep the y-axis starting at zero — the S$150k gap between top and bottom is about 18% of the price, and a truncated axis would make it look like a cliff.

The Python idiom that builds `price_by_town` is a *dict comprehension*: it builds a dictionary in one expression. Read it as "for each (town, price) pair in the zipped lists, create a key `town` with value `{"Median Price (S$)": price}`". Dict comprehensions are to dicts what list comprehensions are to lists; you will see them often.

### Step 4: Correlation heatmap

Polars computes the correlation matrix itself; Plotly draws it:

```python
numeric_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "year"]
corr = hdb_clean.select(numeric_cols).corr()   # Polars correlation matrix

fig_heatmap = px.imshow(
    corr.to_numpy(),
    x=corr.columns,
    y=corr.columns,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Pearson Correlation Matrix — HDB Features",
)
fig_heatmap.write_html("hdb_heatmap.html")
```

`DataFrame.corr()` returns the Pearson correlation of every pair of columns as a square DataFrame. `plotly.express` (`px`) is the higher-level Plotly API; `px.imshow` draws any 2-D grid as a heatmap, and it takes the matrix as a NumPy array via `.to_numpy()`. Plotly Express also accepts Polars DataFrames directly — no pandas conversion needed. ModelVisualizer has no correlation-heatmap method (`ModelVisualizer.confusion_matrix` takes true and predicted *labels* for classification, not a matrix), so we drop down to Plotly. That is fine — ModelVisualizer is a convenience wrapper, not a wall.

Read the result. The diagonal is all 1s (every variable correlates with itself). With `RdBu_r`, red is positive and blue is negative. `resale_price` vs `floor_area_sqm` is the darkest red off the diagonal (0.91). `resale_price` vs `price_per_sqm` is moderately red (0.45). `floor_area_sqm` vs `price_per_sqm` is near white (0.05): in this data a bigger flat costs more in total but not more per square metre. And `year` vs everything is white (about 0.00): prices in this synthetic dataset do not trend over time — the same finding as Lesson 1.5, now visible at a glance.

### Step 5: Line chart of annual median price

Pick the top 5 most-transacted towns and plot their annual medians as separate lines:

```python
top_5_towns = (
    district_prices.sort("transaction_count", descending=True)["town"].head(5).to_list()
)

annual = (
    hdb_clean.filter(pl.col("town").is_in(top_5_towns))
    .group_by("year", "town")
    .agg(pl.col("resale_price").median().alias("median_price"))
    .sort("year")
)

years = sorted(annual["year"].unique().to_list())
price_series = {}
for town in top_5_towns:
    town_data = annual.filter(pl.col("town") == town).sort("year")
    lookup = dict(zip(town_data["year"].to_list(), town_data["median_price"].to_list()))
    price_series[town] = [float(lookup[y]) for y in years]

fig_line = viz.training_history(
    metrics=price_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
# training_history plots every series against 1..N — put the real years on the x-axis
fig_line.update_traces(x=years)
fig_line.update_layout(title="Annual Median HDB Price — Top 5 Towns")
fig_line.write_html("hdb_line.html")
```

`training_history` was built for loss curves, so it plots each list against 1, 2, 3, … and calls the x-axis "Epoch" by default. We pass `x_label="Year"` and then replace the x-values with the real years using `fig_line.update_traces(x=years)`; without that the axis would read 1–10, which is a misleading chart. (`lookup[y]` deliberately raises if a town has no sales in some year, rather than silently plotting a zero.)

The chart shows five nearly flat lines, every point between about S$796k and S$874k, wobbling by a few percent from year to year with no direction. There is no divergence and no crossing worth reporting. On real resale data this chart would show prices rising after 2020; on this synthetic file, the honest reading is "no trend" — and a flat line is a finding, not a failed chart.

### Step 6: Stacked bar of flat-type composition

The sixth chart answers a composition question: what mix of flat types does each town sell?

```python
composition = (
    hdb_clean.group_by("town", "flat_type")
    .agg(pl.len().alias("count"))
    .sort("town", "flat_type")
)

fig_stacked = px.bar(
    composition,
    x="town", y="count", color="flat_type",
    title="Flat Type Composition by Town",
)
fig_stacked.write_html("hdb_stacked.html")
```

Plotly Express takes the Polars DataFrame directly. `color="flat_type"` splits each town's bar into coloured segments, one per flat type, stacked on top of each other (Plotly Express stacks coloured bars by default; `barmode="group"` would put them side by side). The total bar height is the town's transaction count, so the chart answers two questions at once: which towns are busiest (bar height) and what they sell (segments).

Stacked bars have a weakness: only the bottom segment sits on a common baseline, so comparing the middle segments across towns is hard. When the question is *proportions* — "which town sells the largest share of 5-room flats?" — use a **100% stacked bar**, where every bar has the same height and each segment shows a share:

```python
composition_pct = composition.with_columns(
    (pl.col("count") / pl.col("count").sum().over("town") * 100).alias("share_pct")
)

fig_pct = px.bar(
    composition_pct,
    x="town", y="share_pct", color="flat_type",
    title="Flat Type Mix by Town (% of each town's sales)",
)
fig_pct.update_layout(yaxis_title="Share of town's sales (%)")
fig_pct.write_html("hdb_stacked_pct.html")
```

`pl.col("count").sum().over("town")` is the Lesson 1.5 window pattern: each row is divided by its own town's total, so every town's segments add up to 100%. This is also the standard chart for survey and Likert-scale data (strongly disagree … strongly agree): one 100% bar per question, segments ordered from negative to positive.

In this dataset the mix is strikingly similar across towns (4-room is between 37% and 44% of sales in every town) — another fingerprint of synthetic data. Saving these HTML files gives you an informal dashboard you can open in a browser; Lesson 1.8 combines charts into a single report.

## Try It Yourself

**Drill 1.** Create a histogram of `price_per_sqm` instead of `resale_price`. Is the shape different from the resale price histogram? Why?

**Drill 2.** Create a scatter plot of `floor_area_sqm` vs `price_per_sqm`. What do you see? Is there a positive or negative trend, and what does the sign tell you about the relationship between flat size and per-unit price?

**Drill 3.** Build a bar chart of the *count* of transactions per flat type (not median price). Use `viz.metric_comparison` with `{flat_type: {"count": n}}`. Which flat type is most common?

**Drill 4.** Create a line chart with one line per flat type showing annual median price. Use the same pattern as Step 5 but group by `(year, flat_type)` instead of `(year, town)`. Remember to put the real years on the x-axis.

**Drill 5.** Build a correlation heatmap for these columns: `resale_price`, `floor_area_sqm`, `price_per_sqm`, `year`, `lease_commence_date`. What is the correlation between `year` and `lease_commence_date`? Does it surprise you?

**Drill 6.** Build a 100% stacked bar of flat-type share by *year* instead of by town (`x="year"`). Does the mix of flat types sold change over the decade?

## Cross-References

- **Lesson 1.7** will use ModelVisualizer for diagnostic charts generated by DataExplorer — the engine produces its own set of standard charts automatically.
- **Lesson 1.8** will combine multiple charts into an end-to-end HTML report.
- **Module 2** will reuse these chart types for feature analysis — distribution plots per feature, correlation heatmaps of engineered features, scatter plots of feature vs target.
- **Module 3** will introduce `training_history` for its original purpose: plotting loss curves during model training.

## Reflection

You should now be able to:

- Choose an appropriate chart type for each of the common data questions.
- Instantiate a `ModelVisualizer` and call its `histogram`, `scatter`, `metric_comparison`, and `training_history` methods with Polars DataFrames, and fix the labels of a repurposed chart.
- Build a correlation heatmap with `df.corr()` and `px.imshow`, and stacked / 100%-stacked bars with `px.bar`.
- Export charts as standalone HTML with `fig.write_html()`.
- Explain the six Gestalt principles (proximity, similarity, closure, continuity, connection, enclosure) in your own words and apply them to critique a chart.
- Identify the most common misleading chart designs (3D, pie charts, dual y-axes, truncated y-axis on bar charts).
- Sample a large dataset before plotting a scatter plot to keep the chart readable.

### Drill answers

1. ```python
   fig = viz.histogram(data=hdb, column="price_per_sqm", bins=40)
   print(hdb["resale_price"].skew(), hdb["price_per_sqm"].skew())   # about 11.4 and 22.1
   ```
   On the raw data the price-per-sqm histogram is *more* skewed (22.1 vs 11.4): dividing the S$9M sales by a small floor area produces values near S$100,000 per sqm, an even longer tail. On `hdb_clean` both are nearly symmetric (0.27 for price per sqm, 0.39 for price). Normalising by area does not remove bad data — it can amplify it.
2. ```python
   fig = viz.scatter(data=hdb_clean.sample(5_000, seed=42), x="floor_area_sqm", y="price_per_sqm")
   ```
   Sample from `hdb_clean`, or the planted rows stretch the axis. There is essentially no trend: the correlation is about 0.05. A flat band means bigger flats cost more in total but not more (or less) per square metre. In real markets small flats often carry a per-sqm premium; this synthetic dataset was generated without one — which you can only learn by looking.
3. ```python
   flat_counts = hdb.group_by("flat_type").agg(pl.len().alias("count")).sort("count", descending=True)
   data = {ft: {"count": float(c)} for ft, c in zip(flat_counts["flat_type"].to_list(), flat_counts["count"].to_list())}
   viz.metric_comparison(data).show()
   ```
   4 ROOM is most common with 20,299 of 50,150 transactions (about 40%). Set `fig.update_layout(xaxis_title="Flat type", yaxis_title="Transactions")` so the axes do not read "Model"/"Score".
4. Same pattern as Step 5 with a `(year, flat_type)` group_by, a dict of one list per flat type, and `fig.update_traces(x=years)`. You will see six flat, well-separated lines (from about S$340k for 2-room to about S$1.5M for multi-generation): flat type matters a great deal, year does not.
5. The correlation between `year` and `lease_commence_date` is about 0.002 — effectively zero. In real resale data you would expect a positive correlation (later sales include more recently completed flats), so this *should* surprise you: it is another sign that the course dataset was generated column by column rather than recorded. `lease_commence_date` is also uncorrelated with price (about −0.01), which a real market would not show. A heatmap is a fast way to check whether the relationships you expect are actually in the data.
6. ```python
   by_year = (
       hdb_clean.group_by("year", "flat_type").agg(pl.len().alias("count"))
       .with_columns((pl.col("count") / pl.col("count").sum().over("year") * 100).alias("share_pct"))
       .sort("year", "flat_type")
   )
   px.bar(by_year, x="year", y="share_pct", color="flat_type").write_html("mix_by_year.html")
   ```
   Every year's bar looks the same: the flat-type mix is stable across the decade.

---

# Lesson 1.7: Automated Data Profiling

## Why This Matters

Every time you open a new dataset, you do the same things. Check the shape. Check the columns. Check the types. Count the nulls. Look at a few rows. Compute summary statistics. Look for outliers. Check for duplicates. Plot the distributions. Compute correlations. Look for columns that are suspiciously constant or suspiciously unique. This is a rigid, repetitive checklist that should never be done by hand — at least not after you have done it manually a dozen times to know what you are looking at.

`DataExplorer` is the Kailash engine that automates this checklist. It takes a DataFrame, runs the full battery of profile checks in parallel, and returns a structured result you can inspect programmatically or render as an HTML report. More importantly, it emits *alerts*: typed, severity-tagged messages that tell you what looks wrong. An alert with type `"high_skewness"` on column `"fare_sgd"` is a concrete, actionable piece of information — you know exactly what to look at, and you can map it to a standard fix.

This lesson is where you stop doing by hand what the engine can do for you. You will configure `AlertConfig` with thresholds appropriate for your domain, run `DataExplorer.profile()` on a deliberately messy economic-indicators dataset, interpret each alert, and compare two time-period slices of the same data to detect distribution drift. You will also meet `try` / `except` for error handling, and `async` / `await` for the first time in the course.

## Core Concepts

### FOUNDATIONS: Why automate profiling?

The profiling you did manually in Lessons 1.1 through 1.6 works, and there is no substitute for doing it once so you understand what each piece means. But there are two reasons to automate it after the first time.

**Consistency.** A manual checklist is only as reliable as the person running it. Every dataset you miss a check on is a potential bug. An automated profiler runs the same checks on every dataset, so you never forget to look at skewness or cardinality.

**Alerts as a decision layer.** Raw statistics are information; alerts are decisions. A mean of 3.5 is information. An alert saying "column `fare_sgd` has skewness 4.2, above the threshold of 2.0" is a prompt to decide — log-transform, winsorise, or investigate the extreme rows. When you are moving fast through many datasets, the decision layer is what matters. You cannot stop to manually evaluate ten statistics per column for a dataset with fifty columns; the alert layer collapses five hundred statistics into the ten or twenty that actually need attention.

DataExplorer does both: it computes the full statistics (so you can dig into any specific number if needed) and it emits alerts (so you know which numbers to dig into first). The rest of this lesson is about understanding the alerts and trusting them appropriately.

### FOUNDATIONS: The eight alert types

DataExplorer emits exactly eight alert categories — no more. Each has a configurable `AlertConfig` threshold (default shown) and a typical remediation.

| Alert type | What it detects | Typical fix |
|---|---|---|
| `high_nulls` | null fraction above `high_null_pct_threshold` (0.05) | impute, drop rows, or drop column |
| `high_zeros` | zero fraction above `zero_pct_threshold` (0.5) | check whether zeros are real or missing-coded-as-zero |
| `high_skewness` | absolute skewness above `skewness_threshold` (2.0) | log-transform, winsorise, or investigate outliers |
| `high_cardinality` | unique-value ratio above `high_cardinality_ratio` (0.9) — near-unique columns such as IDs and dates | treat as an identifier, bin, or drop |
| `constant` | unique values at or below `constant_threshold` (1) | drop the column (no information) |
| `high_correlation` | a pair of columns with \|r\| above `high_correlation_threshold` (0.9) | drop one of the pair to avoid multicollinearity |
| `duplicates` | duplicate-row fraction above `duplicate_pct_threshold` (0.0, i.e. any duplicate) | `.unique()` |
| `imbalanced` | categorical column whose rarest class is below `imbalance_ratio_threshold` (0.1) | oversample, undersample, or class weights |

Each alert is a plain Python dictionary with four keys: `type` (one of the eight names above), `column` (the column name — or `columns`, a two-element list, for `high_correlation`), `value` (the computed number that crossed the threshold), and `severity` (`"info"` or `"warning"`). There is no message or recommendation string — translating an alert into an action is your job, and the table above is the starting point. Because correlation alerts use `columns` rather than `column`, read the column with `alert.get("column", alert.get("columns"))`.

Two details trip people up. First, the comparisons are *strictly greater than*: a column with exactly 5.0% nulls does **not** fire `high_nulls` at the default 0.05 threshold — and the course CPI file has exactly that (15 of 300 rows null in each CPI column). Second, there is no "outlier" alert. Outliers are reported per column as statistics (`outlier_count`, `outlier_pct`, using the 1.5 × IQR rule) on each column profile; you read them yourself.

### FOUNDATIONS: AlertConfig — tuning thresholds for your domain

Out-of-the-box thresholds are reasonable defaults for "typical tabular ML data", but no dataset is perfectly typical. An economic-indicators dataset with macroeconomic variables has *expected* high correlations (CPI and employment are always correlated); flagging them as problems every time would flood the output with false alarms. A taxi dataset has *expected* high skewness in fare and distance (most rides are short and cheap, a few are long and expensive); a default threshold of 2.0 might flag every column.

`AlertConfig` lets you tune each threshold:

```python
from kailash_ml import AlertConfig

alert_config = AlertConfig(
    high_correlation_threshold=0.95,    # raise from the 0.90 default: only near-perfect collinearity
    high_null_pct_threshold=0.10,       # relax from the 0.05 default: allow up to 10% nulls
    constant_threshold=1,                # flag columns with <= 1 unique value
    high_cardinality_ratio=0.95,        # flag columns where >95% of values are unique
    skewness_threshold=3.0,              # only flag severe skew
    zero_pct_threshold=0.30,             # allow up to 30% zeros
    imbalance_ratio_threshold=0.05,     # flag minority class below 5%
    duplicate_pct_threshold=0.05,       # flag duplicates above 5%
)
```

Every threshold is a *deliberate choice*, not a default. When you configure a profiler for a new domain, you should be able to justify each number. "Why 0.95 for correlation?" — because CPI and unemployment are structurally correlated at around 0.85; I only want to catch *near-perfect* collinearity, which is a sign of data pipeline bugs. "Why 3.0 for skewness?" — because crisis periods (GFC, COVID) create genuine outliers in macro data that would trigger alerts at 2.0 but are real information, not errors.

The thresholds are not set-and-forget. After the first profile run, look at which alerts fired and which did not. If you are getting too many false alarms, relax the relevant threshold. If you are missing problems you can see by eye, tighten them. Tuning AlertConfig is an iterative process, much like tuning any other hyper-parameter.

### FOUNDATIONS: `async` and `await` — just enough to use them

`DataExplorer.profile()` is an *asynchronous* function — a function marked with `async def` that returns a *coroutine* rather than a direct value. You cannot call it the way you call a regular function. Instead, you either:

1. Call it from inside another `async def` function, using `await`:
   ```python
   async def my_function():
       profile = await explorer.profile(df)
       return profile
   ```
2. Run the whole async call chain with `asyncio.run(...)`:
   ```python
   import asyncio
   profile = asyncio.run(my_function())
   ```

`async` functions exist so that Python can run multiple I/O operations in parallel without blocking. If DataExplorer is profiling a dataset with 100 columns, it can run the per-column analyses concurrently and finish faster than sequential execution would. That is the reason for the async API.

For Module 1 you only need to know the recipe:

- Wrap calls to async functions in an `async def` wrapper of your own.
- Use `await` on every async call inside that wrapper.
- Run the wrapper once at the top level with `asyncio.run()`.

One catch: `asyncio.run()` works in a script, but inside Jupyter or Colab an event loop is already running and `asyncio.run()` raises `RuntimeError: asyncio.run() cannot be called from a running event loop`. The course ships synchronous helpers in `shared` that work in both places, so you can profile without writing any async code:

```python
from shared import run_compare, run_profile, run_report

profile = run_profile(df)                          # DataProfile (default thresholds)
profile = run_profile(df, alert_config=alert_config)
diff = run_compare(df_raw, df_clean)               # dict, same as DataExplorer.compare
html = run_report(df, title="My data")             # HTML string, same as DataExplorer.to_html
```

The worked example below writes the async version explicitly, so you see what the helpers hide.

You will meet async again in Module 3 (for async inference servers) and Module 6 (for concurrent API calls to LLMs). For now, treat it as ceremonial boilerplate.

### FOUNDATIONS: `try` / `except` — handling errors

When you run code that might fail, Python's `try` / `except` block lets you catch the error and do something sensible instead of crashing:

```python
try:
    profile = asyncio.run(profile_economic_data())
    print("Profile complete.")
except Exception as exc:
    print(f"Profile failed: {exc}")
    raise
```

Reading this: "try to run the indented block under `try`. If any exception is raised during that block, catch it in the `except` clause. In the except clause, `exc` is the exception object; we print it and then re-raise it with `raise` so the calling code still sees the error."

You should use `try` / `except` when:

- You have a recovery action. You want to retry, fall back to a default, or log and continue.
- You want to provide a better error message than the raw exception would produce. Wrapping a cryptic `KeyError: 'foo'` with "configuration file is missing the 'foo' key — check config.yaml" is much more actionable for the user.

You should *not* use `try` / `except` to silently swallow errors. A bare `except: pass` that hides every error is a bug-incubator — it makes broken code appear to work. Always do something with the exception, even if it is just logging and re-raising.

### THEORY: `DataExplorer.compare` — drift detection

`DataExplorer.compare(df_a, df_b)` profiles two DataFrames separately and then computes column-level deltas between them: mean delta, std delta, null-fraction delta and unique-count delta for every shared column. It returns a plain dictionary.

This is the foundation of *drift detection*. If you trained a model on last year's data and the distribution of incoming data has shifted, your model's predictions may be miscalibrated. Comparing a baseline profile (training data) with a current profile (production data) is the standard way to catch drift early. In Module 4 you will meet `DriftMonitor`, the Kailash engine dedicated to this problem; `compare` is its conceptual foundation.

The dictionary's `column_deltas` entry is a list with one delta dictionary per shared column (each delta is B minus A):

```text
[
    {"column": "cpi_all_items", "dtype": "Float64", "mean_delta": 68.4, "std_delta": ...,
     "null_pct_delta": ..., "unique_count_delta": ...},
    ...
]
```

Sort by `abs(mean_delta)` to surface the columns with the biggest distribution shifts. But note that `mean_delta` is in each column's own units, so a column measured in dollars will always out-shift one measured in percentage points. To compare shifts across columns fairly, divide each delta by the column's mean or standard deviation in period A. The other keys of the dictionary are `profile_a`, `profile_b` (full profiles), `shape_comparison` (`{"rows_a", "rows_b", "cols_a", "cols_b"}`), `shared_columns`, `missing_in_a` and `missing_in_b`.

### FOUNDATIONS: Spearman vs Pearson correlation

DataExplorer computes both Pearson (`profile.correlation_matrix`) and Spearman (`profile.spearman_matrix`) correlations. Pearson you already know — it measures *linear* relationships between two variables. Spearman measures *monotonic* relationships: any relationship where y always increases (or always decreases) as x increases, regardless of whether the increase is linear.

The Spearman correlation is the Pearson correlation of the *ranks* of the values. To compute it: rank each value in column A, rank each value in column B, then compute Pearson on the two rank columns. If the ranks agree (both columns rank observations in the same order), Spearman is 1. If the ranks are opposite, -1. If the ranks are unrelated, 0.

Why it matters: two variables can have a strong monotonic relationship but a weak Pearson correlation if the relationship is non-linear. CPI vs GDP is often monotonic but curvilinear. Spearman catches the relationship; Pearson does not.

The rule: if you are screening for *any* dependency, use Spearman. If you specifically need a linear relationship (for linear regression assumptions), use Pearson. DataExplorer reports both, which is the conservative choice.

## The Kailash Engine: DataExplorer — full API

```python
import asyncio

import polars as pl
from kailash_ml import AlertConfig, DataExplorer

df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 9.0]})
df_a, df_b = df.head(2), df.tail(2)
alert_config = AlertConfig(high_null_pct_threshold=0.10)


async def tour() -> None:
    explorer = DataExplorer(alert_config=alert_config)   # alert_config is optional

    # Main profiling call — async
    profile = await explorer.profile(df)

    # Inspect results (plain attributes — no await needed)
    print(profile.n_rows, profile.n_columns)     # ints
    print(profile.duplicate_count, profile.duplicate_pct)
    print(profile.type_summary)       # dict of inferred type -> column count
    print(profile.alerts)             # list of alert dicts (type, column/columns, value, severity)
    print(profile.correlation_matrix) # Pearson, as a dict of dicts
    print(profile.spearman_matrix)    # Spearman, as a dict of dicts

    # Per-column fields (profile.columns is a list of ColumnProfile objects)
    for col in profile.columns:
        print(
            col.name,
            col.inferred_type,        # "numeric" | "categorical" | "boolean" | "constant" | "id" | "text"
            col.mean, col.std,        # numeric columns only (None otherwise)
            col.min_val, col.max_val,
            col.null_count, col.null_pct, col.unique_count,
            col.skewness, col.kurtosis,
            col.outlier_count, col.outlier_pct,   # 1.5 x IQR rule — a statistic, not an alert
        )

    # Compare two datasets — returns a dict
    comparison = await explorer.compare(df_a, df_b)
    print(comparison["shape_comparison"])   # {"rows_a": ..., "rows_b": ..., "cols_a": ..., "cols_b": ...}
    print(comparison["shared_columns"])     # list of column names
    print(comparison["column_deltas"])      # list of per-column delta dicts

    # Generate HTML report
    report_html = await explorer.to_html(df, title="My Dataset")
    with open("report.html", "w") as f:
        f.write(report_html)

    # Generate individual chart figures
    vis_report = await explorer.visualize(df)
    for name, fig in vis_report.figures.items():
        fig.write_html(f"{name}.html")


asyncio.run(tour())
```

The four engine methods — `profile`, `compare`, `to_html`, `visualize` — are async: they read and compute over the data, so you `await` them. Everything you read from the returned profile object is a plain attribute, so no `await` is needed. (`await` is only legal inside an `async def`, which is why the tour is wrapped in one.)

## Worked Example: Profiling Singapore Economic Indicators

### Step 1: Load three messy time series

```python
from __future__ import annotations

import asyncio

import polars as pl
from kailash_ml import DataExplorer
from kailash_ml.engines.data_explorer import AlertConfig

from shared import MLFPDataLoader

loader = MLFPDataLoader()

cpi = loader.load("mlfp01", "sg_cpi.csv")           # Monthly CPI
employment = loader.load("mlfp01", "sg_employment.csv")  # Quarterly labour stats
fx_rates = loader.load("mlfp01", "sg_fx_rates.csv")      # Daily SGD exchange rates

print(f"CPI: {cpi.shape}")
print(f"Employment: {employment.shape}")
print(f"FX: {fx_rates.shape}")
```

Three datasets at three different frequencies. The challenge is that they do not share a common time grain.

### Step 2: Normalise the date columns

Each dataset uses a different date format. CPI uses a mix of `"01/2000"`, `"2000-02"`, and `"201108"`. Employment uses quarter strings like `"2000 Q1"`. FX rates use ISO date strings. Normalise each to a monthly date:

```python
cpi = cpi.with_columns(
    pl.col("date")
    .str.replace(r"^(\d{2})/(\d{4})$", "$2-$1-01")
    .str.replace(r"^(\d{4})(\d{2})$", "$1-$2-01")
    .str.replace(r"^(\d{4})-(\d{2})$", "$1-$2-01")
    .str.to_date("%Y-%m-%d")
    .alias("date")
)


def quarter_to_date(q_str: str) -> str:
    parts = q_str.split()
    year = parts[0]
    q = int(parts[1][1])
    month = {1: "01", 2: "04", 3: "07", 4: "10"}[q]
    return f"{year}-{month}-01"


employment = employment.with_columns(
    pl.col("quarter").map_elements(quarter_to_date, return_dtype=pl.String)
    .str.to_date("%Y-%m-%d")
    .alias("date")
)

if fx_rates["date"].dtype == pl.String:
    fx_rates = fx_rates.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
```

A few new things here. `.str.replace(pattern, replacement)` applies a regex substitution to every value in a string column. The patterns with `$1` and `$2` are backreferences to the capture groups from the regex. `.map_elements(fn, return_dtype=...)` applies an arbitrary Python function to each element of a column — it is slower than a native Polars expression because Python is involved per-element, but it is the escape hatch when you need custom logic. We use it for `quarter_to_date` because writing that conversion as a pure Polars expression is awkward.

### Step 3: Build a common monthly spine and align everything

```python
cpi = cpi.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
employment = employment.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))

date_range = pl.date_range(
    cpi["month_date"].min(),
    cpi["month_date"].max(),
    interval="1mo",
    eager=True,
)
monthly_spine = pl.DataFrame({"month_date": date_range})

employment_monthly = (
    monthly_spine.join(employment.drop("date"), on="month_date", how="left")
    .sort("month_date")
    .with_columns([
        pl.col(c).forward_fill()
        for c in employment.columns
        if c not in ("date", "month_date")
    ])
)

fx_monthly = (
    fx_rates.with_columns(pl.col("date").dt.truncate("1mo").alias("month_date"))
    .group_by("month_date")
    .agg([pl.col(c).mean() for c in fx_rates.columns if c != "date"])
    .sort("month_date")
)

economic = (
    cpi.join(employment_monthly, on="month_date", how="left", suffix="_emp")
    .join(fx_monthly, on="month_date", how="left", suffix="_fx")
    .sort("month_date")
)

print(economic.shape)
```

Three alignment techniques in one step:

- `dt.truncate("1mo")` truncates a date to the first of its month, giving every record a canonical month date.
- `pl.date_range(start, end, interval="1mo")` creates a complete sequence of monthly dates, used as a "spine" to ensure every month is present even if some source datasets have gaps.
- `.forward_fill()` replaces NULLs with the most recent non-null value — the standard way to upsample quarterly data to monthly.
- `group_by("month_date").agg([pl.col(c).mean() for c in ...])` aggregates the daily FX rates into monthly means.

This is a combination of joins from Lesson 1.4 and aggregation from Lesson 1.3 — everything you have learned so far converging on a single messy real-world problem.

### Step 4: Configure AlertConfig

```python
alert_config = AlertConfig(
    high_correlation_threshold=0.95,
    high_null_pct_threshold=0.10,
    constant_threshold=1,
    high_cardinality_ratio=0.95,
    skewness_threshold=3.0,
    zero_pct_threshold=0.30,
    imbalance_ratio_threshold=0.05,
    duplicate_pct_threshold=0.05,
)
```

Each number reflects the nature of economic data: structurally correlated series (so high correlation threshold), edge-null tolerance for forward-filled columns (so 10% null threshold), crisis-period outliers (so skewness threshold 3.0 ignores milder asymmetry).

### Step 5: Run the profiler

```python
async def profile_economic_data():
    explorer = DataExplorer(alert_config=alert_config)
    profile = await explorer.profile(economic)

    print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
    print(f"Duplicates: {profile.duplicate_count} ({profile.duplicate_pct:.1%})")

    print(f"\n--- Alerts ({len(profile.alerts)}) ---")
    for alert in profile.alerts:
        col = alert.get("column", alert.get("columns", "N/A"))
        print(f"[{alert['severity'].upper()}] {alert['type']}: {col} = {alert.get('value', 'N/A')}")

    return profile


profile = asyncio.run(profile_economic_data())
```

Expected output (abridged):

```text
Rows: 300, Columns: 15
Duplicates: 0 (0.0%)

--- Alerts (18) ---
[INFO] high_cardinality: date = 1.0
[INFO] high_cardinality: month_date = 1.0
[WARNING] high_nulls: usd_sgd = 0.8
[WARNING] high_nulls: eur_sgd = 0.8
[WARNING] high_nulls: gbp_sgd = 0.8
[WARNING] high_nulls: jpy_sgd = 0.8
[WARNING] high_correlation: ['cpi_all_items', 'cpi_food'] = 0.9951132968970791
...
```

Read each group and decide:

- **`high_cardinality` on `date` and `month_date` (info).** Every row has a unique date — that is what a time index is. Expected; no action.
- **`high_nulls` on the four FX columns (80%).** The FX file starts in January 2020, but the spine runs from 2000. 240 of 300 months have no FX rate. This alert was *created by the merge*, not by the source data — exactly the kind of thing profiling is for. Either restrict FX analysis to 2020 onwards, or accept the nulls knowingly.
- **12 `high_correlation` pairs (|r| from 0.956 to 0.998).** The four CPI series and median income all rise steadily over 25 years, so they correlate almost perfectly. The two pairs with `gbp_sgd` are computed on only the 60 months where FX exists, and are more likely two trends than a relationship. For modelling, keep one CPI series and treat the rest as redundant.

Notice what did *not* fire. Each CPI column has 15 nulls out of 300 rows — exactly 5% — and the rule is "greater than", so even the default threshold would not flag it. You only find those nulls by reading the column profiles (`col.null_pct`). And no skewness alert fired: the tuned threshold of 3.0 is above every column's skewness (the largest is about −2.2, for `employment_rate`).

### Step 6: Compare two time periods

```python
async def compare_periods():
    explorer = DataExplorer(alert_config=alert_config)

    covid_cutoff = pl.date(2020, 3, 1)
    pre_covid = economic.filter(pl.col("month_date") < covid_cutoff)
    during_covid = economic.filter(pl.col("month_date") >= covid_cutoff)

    comparison = await explorer.compare(pre_covid, during_covid)

    deltas = sorted(
        comparison["column_deltas"],
        key=lambda d: abs(d.get("mean_delta", 0)),
        reverse=True,
    )
    print("Top 10 column mean shifts:")
    for d in deltas[:10]:
        print(f"  {d['column']}: mean Δ={d.get('mean_delta', 0):+,.3g}")

    return comparison


comparison = asyncio.run(compare_periods())
```

Expected output (first five lines):

```text
Top 10 column mean shifts:
  median_income: mean Δ=+1.17e+03
  labour_force: mean Δ=+892
  cpi_food: mean Δ=+72
  cpi_housing: mean Δ=+71.5
```

Read this carefully before concluding "COVID changed income most". The deltas are in each column's own units — dollars for `median_income`, thousands of people for `labour_force`, index points for CPI — so the ranking mostly reflects units and long-run growth (both periods span many years of steadily rising series). The change that matters for a labour market is further down: `employment_rate` fell by 1.8 points and `unemployment_rate` rose by 1.28 points, both in percentage points. Before ranking shifts across columns, scale them (for example, divide each `mean_delta` by the column's standard deviation in period A). The comparison is a drift detector: any column that shifted far relative to its own spread is a candidate for investigation, model retraining, or alerting downstream consumers. Note the `return comparison` — the function hands the result back so `main()` can use it in Step 8.

### Step 7: Generate an HTML report

```python
async def generate_report():
    explorer = DataExplorer(alert_config=alert_config)
    report_html = await explorer.to_html(economic, title="Singapore Economic Indicators")
    with open("economic_profile.html", "w") as f:
        f.write(report_html)


asyncio.run(generate_report())
```

Open `economic_profile.html` in a browser. You will see a full dashboard: summary statistics, per-column profiles, alert list, correlation heatmap, distribution plots. This is the same output you would produce manually — but in one call instead of fifty.

### Step 8: Wrap everything in try/except

```python
async def main():
    profile = await profile_economic_data()
    comparison = await compare_periods()
    await generate_report()
    return profile, comparison


try:
    profile, comparison = asyncio.run(main())
    print("All profiles complete.")
except Exception as exc:
    print(f"Profiling failed: {exc}")
    raise
```

Bundling the three async calls into one `main()` coroutine and running it with a single `asyncio.run` avoids creating multiple event loops. In Jupyter or Colab, replace the `asyncio.run(...)` calls with the `shared` helpers (`run_profile`, `run_compare`, `run_report`) shown earlier. The `try` / `except` wrapper catches any unexpected error and prints a readable message before re-raising so the traceback is still shown.

## Try It Yourself

**Drill 1.** Run DataExplorer with the *default* AlertConfig (no custom thresholds). How many alerts do you get compared to the tuned config? Which alerts are new?

**Drill 2.** Tighten the correlation threshold to 0.80 and rerun. Look at the new correlation alerts. Are any of them genuinely surprising, or are they all structurally expected?

**Drill 3.** Compare pre-2008 data with post-2008 data (GFC cutoff). Which column has the largest `mean_delta`? Interpret the shift in domain terms.

**Drill 4.** Write your own alert interpreter function that takes an alert dict and returns a plain-English sentence describing the issue and a recommended fix. Use it to format the alert output.

**Drill 5.** Profile just the FX rates table (before merging). Compare its alerts with the merged economic table's. Which alerts appear only after the merge, and why?

**Drill 6.** Write a small cleaning step for the economic table that you can justify from the profile (for example, keep one CPI series, or restrict to months where FX exists). Then use `run_compare(economic, economic_clean)` (or `explorer.compare`) to show what changed: rows, columns, and the null fractions.

## Cross-References

- **Lesson 1.8** uses DataExplorer as the first step in a full cleaning pipeline: profile → decide → clean → re-profile.
- **Module 4** introduces `DriftMonitor`, which is DataExplorer's `compare` method productionised — it runs in a streaming manner and raises alerts when incoming data drifts from the training distribution.
- **Module 2** will use profile outputs to guide feature selection: drop high-null columns, transform high-skewness ones, bin high-cardinality ones.

## Reflection

You should now be able to:

- Explain what DataExplorer does and why automating profiling is useful.
- List the eight alert types and give an example remediation for each.
- Configure `AlertConfig` with domain-appropriate thresholds and justify each choice.
- Call `DataExplorer.profile()` inside an `async def` wrapper and run it with `asyncio.run`.
- Interpret alert objects (type, severity, column, value) and map them to cleaning actions.
- Use `DataExplorer.compare()` to detect distribution drift between two DataFrames.
- Generate an HTML report with `DataExplorer.to_html()`.
- Use `try` / `except` to catch and re-raise errors with added context.

### Drill answers

1. The default config produces 29 alerts against 18 for the tuned config. The 11 new ones: `high_cardinality` on the four CPI columns (their default threshold is 0.9 and about 94% of CPI values are unique — expected for a continuous index), `high_skewness` on `employment_rate` (−2.25) and `labour_force` (−2.13) because the default skewness threshold is 2.0, and five extra `high_correlation` pairs between 0.90 and 0.95 (for example `employment_rate`/`unemployment_rate` at −0.94). The default correlation threshold is 0.90. The FX `high_nulls` alerts appear in both.
2. At 0.80 you get 23 correlation alerts (35 alerts in total). The new pairs mostly involve `usd_sgd` (with CPI series, median income and `gbp_sgd`). None is surprising: they are trending series computed over only the 60 months where FX exists. The threshold is too tight for macro data — it flags shared trends, not data problems.
3. Use `pl.date(2008, 9, 1)` as the cutoff. The largest raw `mean_delta` is `median_income` (about +1,117), then `labour_force` (+642), then the CPI series (+61 to +62). As in Step 6, these are steadily growing series measured in large units, so the raw ranking says "later years are higher", not "the GFC caused this". Scale by each column's standard deviation before you interpret a shift as a regime change.
4. ```python
   def interpret(alert: dict) -> str:
       t = alert["type"]
       col = alert.get("column", alert.get("columns", "N/A"))
       v = alert.get("value", "N/A")
       templates = {
           "high_nulls": lambda: f"{col} has {v:.1%} missing — impute or drop",
           "high_skewness": lambda: f"{col} skew={v:.2f} — log-transform or winsorise",
           "constant": lambda: f"{col} has no variance — drop column",
           "high_correlation": lambda: f"{col} |r|={abs(v):.2f} — consider dropping one",
           "high_cardinality": lambda: f"{col} is {v:.0%} unique — an ID or date? bin or drop",
       }
       return templates[t]() if t in templates else f"{t} on {col}: {v}"
   ```
   The `lambda:` wrappers build each sentence only for the alert type that matched — without them, every f-string would be evaluated on every call, and `{v:.1%}` would fail whenever `v` is not a number.
5. The raw FX table fires a single alert: `high_cardinality` on `date` (every day is unique). Its nulls are 39 of 1,305 rows (3%), below the threshold. After the merge, the four FX columns fire `high_nulls` at 80%, because daily FX from 2020 onwards was aligned to a monthly spine starting in 2000. The merge, not the source, created those nulls — always profile both the inputs and the joined result.
6. One defensible answer:
   ```python
   from shared import run_compare

   economic_clean = (
       economic.filter(pl.col("month_date") >= pl.date(2020, 1, 1))
       .drop("cpi_food", "cpi_transport", "cpi_housing", "date", "quarter")
   )
   diff = run_compare(economic, economic_clean)
   print(diff["shape_comparison"])     # 300 -> 60 rows, 15 -> 10 columns
   for d in diff["column_deltas"]:
       if d["column"].endswith("_sgd"):
           print(d["column"], d["null_pct_delta"])   # about -0.8: the FX nulls are gone
   ```
   The justification comes from the profile: the extra CPI series are near-duplicates (r > 0.97) of `cpi_all_items`, and the FX columns are only meaningful from 2020. The comparison is the proof that the fix did what you claimed.

---

# Lesson 1.8: Data Pipelines and End-to-End Project

## Why This Matters

Every previous lesson taught one piece of the puzzle. Lesson 1.1 taught you to look at raw data. Lessons 1.2 and 1.3 taught you to filter and aggregate. Lesson 1.4 taught you to join. Lesson 1.5 taught you to compute trends. Lesson 1.6 taught you to plot. Lesson 1.7 taught you to profile. This lesson puts all of it together into a single pipeline: extract → profile → clean → feature-engineer → preprocess → visualise → re-profile. This is the shape of almost every exploratory data analysis project you will ever do, regardless of domain.

You will work on a deliberately messy dataset — a synthetic log of Singapore taxi trips with swapped GPS coordinates, negative fares, trips with zero or negative passengers, trips dated years in the future, fifteen different spellings of four payment methods, and trip IDs reused for different trips. The mess is planted on purpose, but every one of those defects is the kind real data arrives with. Your job is to turn the raw mess into a model-ready dataset without losing signal to the noise and without introducing bugs along the way.

This lesson also introduces `PreprocessingPipeline`, the third Kailash engine in Module 1. Where DataExplorer profiles, PreprocessingPipeline prepares: it imputes missing values, scales numerics, encodes categoricals, and splits rows into train and test sets. It is the bridge between raw data and the model-training steps you will meet in Module 3 — and it has one behaviour you must know about to use it safely.

## Core Concepts

### FOUNDATIONS: The ETL pattern

Every data pipeline, regardless of scale or domain, has the same three stages:

- **Extract.** Get the data from somewhere. Read a file, call an API, query a database, receive a stream.
- **Transform.** Clean it, enrich it, reshape it, compute features. This is where 80% of the work happens.
- **Load.** Put the cleaned data somewhere downstream — a file, a database, a model, a dashboard.

The acronym is ETL (extract-transform-load). Sometimes you see ELT (extract-load-transform), which is the same thing with a different ordering for architectures where you prefer to land raw data first and transform it in the warehouse. The conceptual stages are the same.

For this lesson the pipeline will be:

1. **Extract.** Load `sg_taxi_trips.parquet` from the course data loader (and, in the section below, data from a REST API).
2. **Profile.** Count domain-rule violations and run DataExplorer to identify quality issues.
3. **Clean.** Repair swapped coordinates; drop impossible fares, passengers and dates; normalise payment labels; resolve duplicate IDs; fill nulls whose meaning is known.
4. **Engineer.** Extract hour-of-day, day-of-week, weekend flag, time period, trip duration, distance from the CBD and average speed.
5. **Preprocess.** Hold out test rows, then use PreprocessingPipeline to impute, encode and scale.
6. **Visualise.** Produce diagnostic charts.
7. **Re-profile.** Compare the cleaned data with the original and explain every alert that remains.

Each stage has a clear handoff: the output of one stage is the input of the next. You can re-run any stage independently, which is essential for iteration.

### FOUNDATIONS: Extracting data from a REST API

Files are only one source. Much public data is served by web APIs — and the most common kind is a **REST API**: you send an HTTP request to a URL, and the server sends back data, almost always as **JSON** (JavaScript Object Notation — nested dictionaries and lists, which map directly onto Python `dict` and `list`).

The two request types you need:

- **GET** — "give me data". Everything that describes what you want goes into **query parameters**, the `?key=value&key2=value2` part of a URL.
- **POST** — "here is data". You send a body (usually JSON) — for example, a record to store or a batch of inputs to score.

Python's `httpx` library (installed with the course) makes both a single call. Singapore's OneMap service has a public search endpoint that needs no account:

```python
import httpx
import polars as pl

response = httpx.get(
    "https://www.onemap.gov.sg/api/common/elastic/search",
    params={"searchVal": "Tampines", "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1},
    timeout=10,
)
response.raise_for_status()      # stop with an error on a 4xx / 5xx status
data = response.json()           # JSON text -> Python dict
print(data["found"], "matches;", len(data["results"]), "on this page")
```

`params=` builds the query string for you (`?searchVal=Tampines&returnGeom=Y&…`). `timeout=10` stops the program waiting forever on a dead server. `raise_for_status()` turns an HTTP error — 404 not found, 500 server error — into a Python exception instead of letting you parse an error page as if it were data. The response is a dictionary: `found` (how many matches in total), `pageNum` and `totalNumPages` (results come back one page at a time), and `results`, a list of one dictionary per match.

A list of flat dictionaries is exactly what `pl.DataFrame` accepts:

```python
locations = pl.DataFrame(data["results"]).select(
    pl.col("SEARCHVAL").alias("place"),
    pl.col("POSTAL").alias("postal_code"),
    pl.col("LATITUDE").cast(pl.Float64).alias("lat"),
    pl.col("LONGITUDE").cast(pl.Float64).alias("lng"),
)
print(locations.head(3))
```

Note the `.cast(pl.Float64)`: this API returns coordinates as *strings* (`"1.3433…"`), so you must convert them before any arithmetic. Check the types of everything an API gives you — the same discipline as checking a file's schema.

A POST looks the same, with the payload in `json=` instead of `params=`. Here the public test service httpbin.org simply echoes back what it received:

```python
reply = httpx.post("https://httpbin.org/post", json={"town": "TAMPINES", "flat_type": "4 ROOM"}, timeout=10)
reply.raise_for_status()
print(reply.json()["json"])      # {'flat_type': '4 ROOM', 'town': 'TAMPINES'}
```

Three habits make API extraction reliable. Always set a timeout. Always check the status before parsing. And wrap the call in `try` / `except httpx.HTTPError` with a clear fallback — a saved copy of the last good response, or a clear error — because a pipeline that depends on a network call will eventually meet a day when the network is down. (The live results above change as the service's data changes; treat the counts as examples.)

### FOUNDATIONS: Project structure — modules and imports

A notebook or a single 900-line script is fine for exploring. A pipeline you will re-run, test and hand to a colleague should be split into **modules**: separate `.py` files, each owning one stage.

```text
taxi_pipeline/
├── extract.py       # load_taxi_data(), fetch_locations()
├── transform.py     # clean_taxi_data(), add_features(), preprocess()
├── visualise.py     # create_charts()
├── report.py        # write_report()
└── main.py          # the orchestrator: calls the stages in order
```

Each file defines functions; `main.py` imports and calls them:

```python
# main.py
from extract import load_taxi_data
from report import write_report
from transform import add_features, clean_taxi_data
from visualise import create_charts


def main() -> None:
    raw = load_taxi_data()
    clean = add_features(clean_taxi_data(raw))
    charts = create_charts(clean)
    write_report(raw, clean, charts)


if __name__ == "__main__":
    main()
```

`from transform import clean_taxi_data` works because `transform.py` sits in the same folder: Python treats every `.py` file as a module you can import by its filename. The `if __name__ == "__main__":` line means "run `main()` only when this file is executed directly (`python main.py`), not when another file imports it" — so you can import `main.py`'s functions into a test or a notebook without triggering the whole pipeline. The payoff of this structure: each stage can be tested on its own, re-run on its own, and replaced without touching the others. Lesson 1.8's Drill 5 asks you to build exactly this.

### FOUNDATIONS: Null handling

Real datasets have missing values. The three decisions you have to make are:

**1. How to detect a null.** Polars treats `null` (the typed missing marker) distinctly from NaN (not-a-number, used for undefined float results) and from empty strings. When loading from CSV, missing values appear as nulls; when reading from some other sources, they may appear as empty strings or a sentinel value like `-999`. Always check the null count per column after loading to know what you are dealing with.

**2. Whether to impute, fill with a meaning, or drop.** Sometimes a null *has* a meaning: in the taxi log, a missing tip means no tip was given, so `0.0` is not an estimate but the truth. If the column is critical and the nulls are a small fraction, impute — fill with a sensible estimate (median for numeric, mode for categorical). If the column is not critical, drop it. If the nulls are concentrated in specific rows (and those rows are unusable for other reasons too), drop the rows.

**3. Which imputation strategy.** Median is the safe default for numeric columns — it is robust to outliers. Mean is acceptable for symmetric distributions. Mode is the default for categoricals. PreprocessingPipeline also offers `imputation_strategy="knn"`, which estimates a missing value from the most similar rows — but for Module 1 the median is enough.

The Polars methods:

- `pl.col("x").is_null()` — Boolean column, True where the value is null.
- `pl.col("x").fill_null(value)` — replace nulls with `value`.
- `pl.col("x").fill_null(strategy="forward")` — replace nulls with the previous non-null.
- `df.drop_nulls(subset=["x", "y"])` — drop rows where any of the named columns is null.

### FOUNDATIONS: Domain-aware cleaning

Generic rules ("drop negative values") are not always right. Domain-aware rules are.

For a Singapore taxi dataset:

- **GPS bounding box.** Singapore lies within latitude `[1.15, 1.47]` and longitude `[103.60, 104.05]`. A point outside it is a GPS error — but look before you drop: a "latitude" of 103.8 paired with a "longitude" of 1.35 is a Singapore point with the two fields swapped. That fix is unambiguous, so repair it. Drop only what cannot be repaired.
- **Fares.** Negative fares are impossible, and zero fares are not useful training examples. Drop them.
- **Passengers.** A paid trip has at least one passenger. Drop counts below 1.
- **Dates.** A log extracted at the end of 2024 cannot contain a trip in 2027. Drop pickups after the extraction date.
- **Categories.** Fifteen spellings of four payment methods (`"CASH"`, `"cash"`, `"Cash Payment"`, …) are one data-entry problem. Normalise them to canonical labels.
- **Keys.** A `trip_id` should identify exactly one trip. If two *different* trips share an ID, you cannot tell which one owns it — drop both.
- **Speed.** After computing duration, you can check distance against time. A whole-trip average above 120 km/h is impossible in Singapore (the expressway speed limit is 90 km/h); one below 2 km/h is slower than walking. Either way, distance or time was recorded wrongly.

Each rule embeds domain knowledge. Generic outlier detection (like "drop values more than 3σ from the mean") would not know that Singapore's GPS box is what it is, or that 120 km/h is the hard ceiling. Always encode the domain knowledge you have; never rely on generic rules alone.

### FOUNDATIONS: Feature engineering for temporal data

Raw timestamps are useless to most ML models — the model cannot directly learn "Tuesday 8 AM is different from Saturday 8 AM". You have to decompose the timestamp into features the model can exploit:

- **Hour of day.** `pickup_datetime.dt.hour()`. An integer 0–23.
- **Day of week.** `pickup_datetime.dt.weekday()`. In Polars this is **1–7, Monday = 1, Sunday = 7** (the ISO convention). Friday is 5. Many other tools count 0–6, so check before you write a weekend rule.
- **Day of month, month of year.** For seasonal effects.
- **Is weekend.** A Boolean derived from day of week: `dt.weekday() >= 6` (Saturday and Sunday). Writing `>= 5` would silently include Friday.
- **Time period** (morning peak, evening peak, off-peak, late night). A categorical derived from hour.
- **Duration.** Subtracting two datetime columns gives a Duration; `.dt.total_seconds()` turns it into a number.

The hour-of-day decomposition is crucial for demand modelling. In real cities, trip volume has a strong daily rhythm — peaks at the morning and evening commute, a trough in the middle of the night — and a model without an hour feature would treat 8 AM and 3 AM trips as equivalent. Whether *this* dataset has such a rhythm is something you check in the worked example rather than assume.

### FOUNDATIONS: Feature engineering for spatial data

Raw latitude and longitude are also not great features. A few more useful derivatives:

- **Haversine distance between two points.** The great-circle distance on a sphere — a direct measure of trip length when you have both pickup and dropoff positions.
- **Distance from a reference point** (city centre, airport, MRT station). "How far is the pickup from Raffles Place?" captures "is this a CBD trip?" — and needs only one position.
- **Bearing.** The compass direction between two points. Sometimes predictive (airport-bound trips head east).
- **Spatial binning.** Divide the city into a grid and turn each point into a grid cell ID.

The course taxi log has only a pickup position (plus a recorded `distance_km`), so the worked example uses distance from the CBD. The haversine formula is worth implementing directly once, because it is a recurring pattern in any geospatial pipeline. Here it is in pure Polars, measuring each pickup's distance to Raffles Place:

```python
import math

import polars as pl

CBD_LAT, CBD_LNG = 1.2840, 103.8514   # Raffles Place
_RAD = math.pi / 180

trips = pl.DataFrame({"pickup_latitude": [1.3521, 1.2840], "pickup_longitude": [103.8198, 103.8514]})
trips = trips.with_columns(
    (
        2 * 6371  # Earth radius in km
        * (
            ((pl.col("pickup_latitude") - CBD_LAT) * _RAD / 2).sin().pow(2)
            + math.cos(CBD_LAT * _RAD)
            * (pl.col("pickup_latitude") * _RAD).cos()
            * ((pl.col("pickup_longitude") - CBD_LNG) * _RAD / 2).sin().pow(2)
        ).sqrt().arcsin()
    ).alias("km_from_cbd")
)
print(trips)   # about 8.35 km for the first point, 0.0 for Raffles Place itself
```

The formula is the haversine great-circle distance:

$$d = 2r \arcsin\left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)$$

where $\phi$ are latitudes, $\lambda$ are longitudes (both in radians), $\Delta \phi$ and $\Delta \lambda$ are the differences, and $r = 6371$ km is Earth's mean radius. For country-scale distances like Singapore's this is accurate to within 0.5% — more than sufficient for feature engineering. For sub-metre precision you would want the more complex Vincenty formula, but you will not need it.

### FOUNDATIONS: PreprocessingPipeline

`PreprocessingPipeline` automates the final steps before model training:

- **Impute** remaining nulls (median, mean or KNN for numeric; mode for categorical).
- **Encode** categorical columns (one-hot or ordinal).
- **Scale** numeric features (standardise to mean 0, standard deviation 1).
- **Infer** the task type (regression if the target is continuous, classification if categorical).
- **Split** the rows it was given into `train_data` and `test_data`.

The call:

```python
from kailash_ml import PreprocessingPipeline

train_rows = pl.DataFrame({"distance_km": [2.0, 5.5, 9.1, 3.3, 12.0], "payment_type": ["Cash", "Card", "Card", "NETS", "Cash"], "fare_sgd": [6.1, 9.8, 14.2, 7.0, 18.5]})
new_rows = pl.DataFrame({"distance_km": [4.0], "payment_type": ["Card"], "fare_sgd": [8.4]})

pipeline = PreprocessingPipeline()      # no constructor arguments
result = pipeline.setup(
    data=train_rows,
    target="fare_sgd",
    normalize=True,
    categorical_encoding="onehot",
    imputation_strategy="median",
)

result.train_data           # Polars DataFrame — setup()'s own split of the rows you passed
result.test_data            # Polars DataFrame
result.numeric_columns      # list of numeric feature columns
result.categorical_columns  # list of categorical feature columns
result.task_type            # "regression" or "classification"
result.original_shape, result.transformed_shape

new_ready = pipeline.transform(new_rows)   # apply the SAME learned rules to new rows
```

The key parameter is `target` — the column you are trying to predict. The pipeline excludes it from the feature set and uses it to infer the task type: classification if the target is a string, categorical or Boolean column, *or a numeric column with 20 or fewer distinct values*; regression otherwise. That rule has a sharp edge — the five-row toy example above has only five distinct fares, so it is inferred as classification, while the 35,000 real fares in the worked example are regression. Always print `result.task_type` and check it is what you meant. `normalize=True` standardises numeric columns to zero mean and unit variance. `categorical_encoding="onehot"` converts each categorical column into a set of binary columns (one per category). `imputation_strategy="median"` fills remaining nulls. `pipeline.transform(df)` re-applies exactly what `setup()` learned — the same medians, categories, means and standard deviations — without refitting.

**The one thing to know: `setup()` fits on everything you give it, and only then splits.** In kailash-ml 2.2.2, `setup()` computes its imputation values, one-hot categories (and, with target encoding, per-category target means) and scaling statistics from *all* rows passed in, and only afterwards divides them into `train_data` and `test_data` (default `train_size=0.8`). So the "test" rows inside `result` have already influenced the transformations — the test-set contamination the next section warns about. The safe pattern, used in the worked example and Exercise 8, is:

1. Hold out your test rows yourself, *before* any fitting (shuffle with a seed, take 20%).
2. Call `setup()` on the training rows only. (Its internal split of those rows is a validation split you can use when choosing models.)
3. Call `pipeline.transform(test_rows)` to prepare the held-out rows with the rules learned from training data.

Never treat `setup(data=all_rows, train_size=0.8)` as a leak-free train/test split.

### ADVANCED: Why standardise numeric features

Linear models (linear regression, logistic regression) and many neural network optimisers are sensitive to feature scale. If one feature has values in the range 0–1 and another in the range 0–1,000,000, gradient-based optimisation is dominated by the large-scale feature: the loss surface is stretched along one direction, so a learning rate small enough to be stable for the big feature makes progress on the small one painfully slow. Standardising both features to mean 0 and std 1 puts them on equal footing.

Tree-based models (decision trees, random forests, gradient boosting) do not need standardisation because they only care about the order of values, not the magnitude. If you are training only tree models, you can skip standardisation. But if you might train any model that uses gradients (as we will in Module 3), standardising is insurance.

The formula for standardisation is:

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ is the column mean and $\sigma$ is the column standard deviation, both computed on the *training* set only. The same $\mu$ and $\sigma$ are then applied to the test set; you do not recompute them. Computing them on data that includes the test rows lets information about the test set leak into training, which makes test scores optimistic. As described above, `PreprocessingPipeline.setup()` computes them on every row it is given — so give it training rows only, and use `transform()` for everything else.

## Worked Example: Taxi Trip Cleaning Pipeline

This walkthrough follows Exercise 8. The dataset, `sg_taxi_trips.parquet`, is a synthetic log of 50,000 Singapore taxi trips, with data-quality problems planted on purpose. Its twelve columns are `trip_id, pickup_datetime, dropoff_datetime, pickup_zone, dropoff_zone, distance_km, fare_sgd, tip_sgd, payment_type, passengers, pickup_latitude, pickup_longitude` — note there is a pickup position but no dropoff position, and the trip length is already given as `distance_km`.

### Step 1: Load and inspect

```python
from __future__ import annotations

import math
from datetime import datetime

import polars as pl
from kailash_ml import AlertConfig, ModelVisualizer, PreprocessingPipeline

from shared import MLFPDataLoader, run_compare, run_profile, run_report

loader = MLFPDataLoader()
taxi_raw = loader.load("mlfp01", "sg_taxi_trips.parquet")

print(f"Shape: {taxi_raw.shape}")
print(taxi_raw.describe())

# The timestamps are stored as strings — parse them before any date arithmetic
taxi_raw = taxi_raw.with_columns(
    pl.col("pickup_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    pl.col("dropoff_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
)

for col in taxi_raw.columns:
    nc = taxi_raw[col].null_count()
    if nc > 0:
        print(f"  {col}: {nc:,} nulls ({nc / taxi_raw.height:.1%})")
```

Read the `describe()` output's min and max rows before anything else. `pickup_latitude` has a maximum near 104 and `pickup_longitude` a minimum near 1.3 — Singapore sits at roughly latitude 1.3, longitude 103.8, so some rows have the two coordinates *swapped*. `fare_sgd` has a negative minimum (−49.97), `passengers` a minimum below 1, and the timestamp columns are strings whose min and max are alphabetical. After parsing, the latest pickup is in 2027 — for a log extracted at the end of 2024. The null scan finds 2,500 missing `pickup_zone` values (5.0%), plus nulls in `dropoff_zone` and `tip_sgd`. The `describe()` output is passive; your job is to decide what to do about each.

### Step 2: Count the problems, then profile

Domain rules first — the profiler cannot know that a fare must be positive:

```python
SG_LAT_MIN, SG_LAT_MAX = 1.15, 1.47
SG_LNG_MIN, SG_LNG_MAX = 103.60, 104.05
DATA_EXTRACT_DATE = datetime(2025, 1, 1)   # the log was extracted at the end of 2024

swapped_gps = taxi_raw.filter(
    pl.col("pickup_latitude").is_between(SG_LNG_MIN, SG_LNG_MAX)
    & pl.col("pickup_longitude").is_between(SG_LAT_MIN, SG_LAT_MAX)
).height
print(f"Swapped GPS:        {swapped_gps:,}")
print(f"Fares <= 0:         {taxi_raw.filter(pl.col('fare_sgd') <= 0).height:,}")
print(f"Passengers < 1:     {taxi_raw.filter(pl.col('passengers') < 1).height:,}")
print(f"Future pickups:     {taxi_raw.filter(pl.col('pickup_datetime') >= DATA_EXTRACT_DATE).height:,}")
print(f"Payment spellings:  {taxi_raw['payment_type'].n_unique()}")
print(f"Colliding trip_ids: {taxi_raw.filter(pl.col('trip_id').is_duplicated()).height:,}")
print(f"Exact duplicates:   {taxi_raw.height - taxi_raw.unique().height:,}")
```

Then the statistical view, using the `run_profile` helper from Lesson 1.7 (it works in scripts and notebooks alike):

```python
alert_config = AlertConfig(
    high_null_pct_threshold=0.02,
    skewness_threshold=2.0,
    high_cardinality_ratio=0.80,
    zero_pct_threshold=0.10,
    high_correlation_threshold=0.90,
)
profile_raw = run_profile(taxi_raw, alert_config)

print(f"Alerts: {len(profile_raw.alerts)}")
for alert in profile_raw.alerts:
    print(f"  [{alert['severity'].upper()}] {alert['type']}: {alert.get('column', alert.get('columns'))}")
```

The domain counts come out as: 250 swapped GPS rows, 1,000 fares at or below zero, 500 rows with fewer than one passenger, 500 pickups dated 2025 or later (up to December 2027), 15 spellings of `payment_type` (`"CASH"`, `"Cash Payment"`, `"cash"`, `"VISA"`, `"GrabPay"`, … for four real methods), 500 rows sharing a `trip_id` — and 0 exact duplicate rows. The profiler, with the thresholds above, raises 15 alerts: `high_nulls` on `pickup_zone`, `dropoff_zone` and `tip_sgd` (78% null — most trips have no tip recorded); `high_cardinality` on `trip_id` and the two timestamps (expected: they are near-unique); `high_skewness` and `high_cardinality` on the coordinates plus a `high_correlation` between latitude and longitude — all symptoms of the 250 swapped rows, which put a "latitude" of 104 into a column of 1.3s; `high_skewness` on `distance_km`; and `imbalanced` on the zone and payment columns.

Compare the two views. The domain rules found the swapped coordinates, the impossible fares and passengers, the trips from the future and the colliding IDs; the profiler found the nulls and the near-unique ID column. Neither view is complete alone. Note especially the last two lines of Step 2: there are no exact duplicate rows, yet hundreds of rows share a `trip_id`. They are different trips given the same ID. `df.unique()` would never find them; only a key check does.

### Step 3: Domain-aware cleaning, one logged step per problem

```python
taxi_clean = taxi_raw.clone()
rows_before = taxi_clean.height
cleaning_log: list[str] = []


def log_step(message: str) -> None:
    cleaning_log.append(message)
    print(message)


# 3a. Repair swapped GPS — the fix is unambiguous, so repair rather than drop
is_swapped = pl.col("pickup_latitude").is_between(SG_LNG_MIN, SG_LNG_MAX) & pl.col(
    "pickup_longitude"
).is_between(SG_LAT_MIN, SG_LAT_MAX)
taxi_clean = taxi_clean.with_columns(
    pl.when(is_swapped).then(pl.col("pickup_longitude")).otherwise(pl.col("pickup_latitude")).alias("pickup_latitude"),
    pl.when(is_swapped).then(pl.col("pickup_latitude")).otherwise(pl.col("pickup_longitude")).alias("pickup_longitude"),
)
log_step(f"GPS: swapped latitude/longitude back in {swapped_gps:,} rows")

# 3b. Drop what cannot be repaired
before = taxi_clean.height
taxi_clean = taxi_clean.filter(
    pl.col("pickup_latitude").is_between(SG_LAT_MIN, SG_LAT_MAX)
    & pl.col("pickup_longitude").is_between(SG_LNG_MIN, SG_LNG_MAX)
    & (pl.col("fare_sgd") > 0)
    & (pl.col("passengers") >= 1)
    & (pl.col("pickup_datetime") < DATA_EXTRACT_DATE)
)
log_step(f"Impossible GPS / fare / passengers / future dates: removed {before - taxi_clean.height:,} rows")

# 3c. Normalise payment_type to four canonical labels
payment_lower = pl.col("payment_type").str.to_lowercase()
taxi_clean = taxi_clean.with_columns(
    pl.when(payment_lower.str.contains("grab")).then(pl.lit("Grab"))
    .when(payment_lower.str.contains("nets")).then(pl.lit("NETS"))
    .when(payment_lower.str.contains("cash")).then(pl.lit("Cash"))
    .when(payment_lower.str.contains("card|visa|mastercard|credit")).then(pl.lit("Card"))
    .otherwise(pl.lit("Other"))
    .alias("payment_type")
)
log_step(f"Payment labels -> {sorted(taxi_clean['payment_type'].unique().to_list())}")

# 3d. trip_id collisions: we cannot tell which trip owns the ID, so drop them all
before = taxi_clean.height
taxi_clean = taxi_clean.filter(~pl.col("trip_id").is_duplicated())
log_step(f"trip_id collisions: removed {before - taxi_clean.height:,} rows")

# 3e. Fill nulls whose meaning is known
taxi_clean = taxi_clean.with_columns(
    pl.col("tip_sgd").fill_null(0.0),            # no tip recorded = no tip
    pl.col("pickup_zone").fill_null("Unknown"),
    pl.col("dropoff_zone").fill_null("Unknown"),
)
log_step("Nulls: tip_sgd -> 0.0, zones -> 'Unknown'")

print(f"Rows: {rows_before:,} -> {taxi_clean.height:,} ({taxi_clean.height / rows_before:.1%} retained)")
```

Expected log: the GPS repair fixes 250 rows; the combined filter removes 1,975 rows (the impossible fares, passengers and future dates overlap slightly, so the total is less than 1,000 + 500 + 500); the payment labels collapse from 15 spellings to `['Card', 'Cash', 'Grab', 'NETS']`; the `trip_id` check removes 456 rows (the colliding IDs that survived the earlier filters); and 47,569 of 50,000 rows (95.1%) remain. A retention rate of 85% or more after a first pass is typical for a log like this; far lower would suggest a systematic upstream problem worth investigating before you clean further.

Two design choices are worth naming. First, *repair when the fix is unambiguous, drop when it is not*: a latitude of 103.8 paired with a longitude of 1.35 is obviously a swapped Singapore point, so we swap it back and keep the trip; a negative fare has no single correct value, so the row goes. Second, every step writes a line to `cleaning_log`, so the pipeline is auditable — anyone can see exactly what was removed and why.

### Step 4: Feature engineering

```python
CBD_LAT, CBD_LNG = 1.2840, 103.8514   # Raffles Place
_RAD = math.pi / 180

taxi_clean = taxi_clean.with_columns(
    # Temporal — Polars weekdays are ISO: Monday = 1 ... Sunday = 7
    pl.col("pickup_datetime").dt.hour().alias("hour_of_day"),
    pl.col("pickup_datetime").dt.weekday().alias("day_of_week"),
    (pl.col("pickup_datetime").dt.weekday() >= 6).alias("is_weekend"),
    # Duration: subtracting two Datetime columns gives a Duration
    ((pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds() / 60).alias(
        "trip_duration_min"
    ),
    # Spatial: haversine distance from the pickup point to the CBD
    (
        2 * 6371
        * (
            ((pl.col("pickup_latitude") - CBD_LAT) * _RAD / 2).sin().pow(2)
            + math.cos(CBD_LAT * _RAD)
            * (pl.col("pickup_latitude") * _RAD).cos()
            * ((pl.col("pickup_longitude") - CBD_LNG) * _RAD / 2).sin().pow(2)
        ).sqrt().arcsin()
    ).alias("km_from_cbd"),
)

taxi_clean = taxi_clean.with_columns(
    pl.when(pl.col("hour_of_day").is_between(7, 9)).then(pl.lit("morning_peak"))
    .when(pl.col("hour_of_day").is_between(17, 20)).then(pl.lit("evening_peak"))
    .when((pl.col("hour_of_day") >= 22) | (pl.col("hour_of_day") <= 5)).then(pl.lit("late_night"))
    .otherwise(pl.lit("off_peak"))
    .alias("time_period"),
    (pl.col("distance_km") / (pl.col("trip_duration_min") / 60)).alias("avg_speed_kmh"),
)

# Derived-feature sanity check: a whole-trip average outside 2-120 km/h is impossible
before = taxi_clean.height
taxi_clean = taxi_clean.filter(pl.col("avg_speed_kmh").is_between(2, 120))
log_step(f"Speed filter (outside 2-120 km/h): removed {before - taxi_clean.height:,} rows")
```

Seven new features: `hour_of_day`, `day_of_week`, `is_weekend`, `trip_duration_min`, `km_from_cbd`, `time_period` and `avg_speed_kmh`. Each embeds a piece of domain knowledge, and each is something a model can learn from, whereas the raw pickup timestamp is not. `is_weekend` uses `>= 6` because Polars numbers weekdays the ISO way (Saturday = 6, Sunday = 7). The speed filter is a consistency check between two columns: a trip that "covers" 20 km in one minute has a wrong distance or a wrong time, even though each value looked plausible on its own.

### Step 5: PreprocessingPipeline — hold out the test rows first

```python
feature_cols = [
    "distance_km", "trip_duration_min", "avg_speed_kmh", "km_from_cbd", "passengers",
    "hour_of_day", "day_of_week", "is_weekend", "time_period", "payment_type",
    "pickup_zone",
]
model_df = taxi_clean.select(feature_cols + ["fare_sgd"])

# Hold out 20% of rows BEFORE fitting anything
shuffled = model_df.sample(fraction=1.0, shuffle=True, seed=42)
n_test = shuffled.height // 5
test_rows, train_rows = shuffled.head(n_test), shuffled.slice(n_test)

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=train_rows,
    target="fare_sgd",
    normalize=True,
    categorical_encoding="onehot",
    imputation_strategy="median",
)
test_ready = pipeline.transform(test_rows)   # same learned rules, never refit

print(f"Task: {result.task_type}")
print(f"Train rows passed to setup(): {train_rows.height:,}  ->  {result.original_shape} -> {result.transformed_shape}")
print(f"Held-out test rows after transform(): {test_ready.shape}")
```

Why split by hand when `setup()` has a `train_size` argument? Because `setup()` learns its imputation medians, one-hot categories and scaling means and standard deviations from **every row you pass it**, and only then splits those rows into `result.train_data` and `result.test_data`. If you passed all 43,934 rows, statistics of the "test" rows would already be baked into the transformations — a quiet form of leakage that makes test scores look better than they will be on genuinely new data. So we hold out 20% (8,786 rows) first, call `setup()` on the other 35,148 only, and apply the learned rules to the held-out rows with `pipeline.transform(test_rows)`, which never refits.

Expected output: `Task: regression` (the target is continuous); `(35148, 12) -> (35148, 53)` — one-hot encoding expands the four categorical columns (`is_weekend`, `time_period`, `payment_type`, `pickup_zone`) into one 0/1 column per category; and the held-out rows come back as `(8786, 53)` with exactly the same columns. Inside `result`, `setup()` has still split the 35,148 training rows 80/20 (`result.train_data` 28,118 rows, `result.test_data` 7,030). Treat that inner split as a *validation* set for choosing models later; your untouched `test_rows` are the real test.

The excluded columns are deliberate. `trip_id` is an identifier, not a feature. The raw timestamps have been turned into hour, weekday and duration. `tip_sgd` is only known *after* the fare is paid, so using it to predict the fare would be *target leakage* — a model that looks brilliant in testing and is useless in practice.

### Step 6: Visualise key patterns

```python
viz = ModelVisualizer()

fig_fare = viz.histogram(taxi_clean, "fare_sgd", bins=60, title="Taxi Fare Distribution (After Cleaning)")
fig_fare.write_html("taxi_fare_distribution.html")

hourly = taxi_clean.group_by("hour_of_day").agg(pl.len().alias("trip_count")).sort("hour_of_day")
fig_hourly = viz.training_history(
    metrics={"Trip Volume": hourly["trip_count"].to_list()},
    x_label="Hour of Day",
    y_label="Number of Trips",
)
fig_hourly.update_traces(x=hourly["hour_of_day"].to_list())   # real hours 0..23, not 1..24
fig_hourly.update_layout(title="Taxi Trip Volume by Hour of Day")
fig_hourly.write_html("taxi_hourly_volume.html")

fig_dist = viz.histogram(taxi_clean, "distance_km", bins=50, title="Trip Distance (km)")
fig_dist.write_html("taxi_distance_distribution.html")
```

Open the charts. The cleaned fares run from S$3.90 to S$50.72 with a median of S$9.64 — no more negative values. The hourly chart is the surprise: it is flat. Every hour of the day has between 1,743 and 1,893 trips, with no morning or evening commute peak. Real taxi demand has a strong daily rhythm; this synthetic log was generated without one, so the honest reading is "no hourly pattern in this data". (Check it by grouping by `time_period` too: average fares are S$9.71–9.74 in every period.) Notice the `update_traces(x=...)` line: without it, `training_history` would label the hours 1–24 instead of 0–23.

### Step 7: Re-profile and compare original vs cleaned

```python
taxi_clean_original_cols = taxi_clean.select(taxi_raw.columns)
profile_clean = run_profile(taxi_clean_original_cols, alert_config)
print(f"Alerts before cleaning: {len(profile_raw.alerts)}")
print(f"Alerts after cleaning:  {len(profile_clean.alerts)}")

comparison = run_compare(taxi_raw, taxi_clean_original_cols)
shape = comparison["shape_comparison"]
print(f"Rows: {shape['rows_a']:,} -> {shape['rows_b']:,}")
fare_a = next(c for c in comparison["profile_a"].columns if c.name == "fare_sgd")
fare_b = next(c for c in comparison["profile_b"].columns if c.name == "fare_sgd")
print(f"fare_sgd min: {fare_a.min_val:.2f} -> {fare_b.min_val:.2f}")

with open("taxi_clean_profile.html", "w") as f:
    f.write(run_report(taxi_clean_original_cols, title="Taxi Trips — Cleaned", alert_config=alert_config))
```

We profile only the original columns, with the same thresholds, so the two alert counts are comparable. Expected output: 15 alerts before and 11 after; rows 50,000 → 43,934; `fare_sgd` minimum −49.97 → 3.90. Read the 11 survivors rather than just counting them. Three are `high_cardinality` on `trip_id` and the two timestamps — correct for identifiers. Two are on `tip_sgd`: after filling nulls with 0, 78% of tips are zero (`high_zeros`) and the column is skewed — true facts about tipping, not errors. The coordinate alerts dropped to plain `high_cardinality` once the swapped rows were repaired. And one alert is *new*: `high_correlation` between `distance_km` and `fare_sgd` (r = 0.91). It was hidden by the negative fares; now that they are gone, the real link between distance and fare shows through. That is the sign of a successful clean — the goal is not zero alerts, it is alerts you can explain.

### Step 8: Pipeline summary

```python
print(f"Stage 1 Load:       {taxi_raw.height:,} rows")
print(f"Stage 2 Profile:    {len(profile_raw.alerts)} alerts")
print(f"Stage 3 Clean:      {taxi_clean.height:,} rows retained ({len(cleaning_log)} logged steps)")
print(f"Stage 4 Engineer:   {len([c for c in taxi_clean.columns if c not in taxi_raw.columns])} new features")
print(f"Stage 5 Preprocess: {train_rows.height:,} train / {test_rows.height:,} held-out test")
print(f"Stage 6 Visualise:  3 charts saved")
print(f"Stage 7 Verify:     {len(profile_clean.alerts)} alerts remaining")
```

A single block that documents the entire pipeline's effect — raw row count, alert counts at entry and exit, clean row count, feature count, train/test split. This is the kind of summary you paste into a pull request or a report. Each number is concrete and auditable.

## Try It Yourself

**Drill 1.** After Step 4, cap extreme fares: compute the 99.9th and the 99.5th percentile of `fare_sgd` and count how many rows lie above each. How many more rows does the tighter cap affect? Would you drop those rows, cap them, or keep them — and why?

**Drill 2.** Using the `km_from_cbd` feature from Step 4, compute the average distance from the CBD for each `time_period`. Do morning-peak trips start closer to the CBD than late-night trips? What does the answer tell you about this dataset?

**Drill 3.** List the columns of `result.train_data` that came from `time_period`, `is_weekend` and `payment_type`. What happened to `hour_of_day` — was it one-hot encoded? What does that imply about how a model will "see" the hour?

**Drill 4.** Rerun Step 5 with `imputation_strategy="mean"` instead of `"median"`. Compare `result.train_data` between the two runs. Explain the result.

**Drill 5.** Split the pipeline into modules as described in "Project structure": an `extract.py` with `load_taxi_data()`, a `transform.py` with `clean_taxi_data()` and `add_features()`, and a `main.py` that calls them and prints the Step 8 summary. Then add a `fetch_locations(search: str) -> pl.DataFrame` function to `extract.py` that calls the OneMap search API and falls back to an empty DataFrame (with the right columns) if the request fails.

## Cross-References

- **Module 2** (Statistical Mastery for Machine Learning and AI Success): will apply the feature-engineering patterns from this lesson at a larger scale using `FeatureEngineer` and store the results in a `FeatureStore`.
- **Module 3** (Supervised ML): will consume `PreprocessingPipeline.result.train_data` directly as input to `TrainingPipeline`. The pipeline boundary you built here is where training takes over.
- **Module 4** (Drift and Monitoring): will schedule `DataExplorer.compare` in a streaming loop as a drift monitor.

## Reflection

You should now be able to:

- Describe the seven stages of an end-to-end data pipeline (load, profile, clean, engineer, preprocess, visualise, re-profile) and give an example of what happens at each.
- Distinguish generic from domain-aware cleaning rules and explain why both are needed.
- Engineer temporal features from a timestamp column using `.dt.hour()`, `.dt.weekday()`, `.dt.month()`.
- Engineer spatial features including the haversine distance using pure Polars expressions.
- Hold out test rows before fitting, use `PreprocessingPipeline.setup()` on the training rows, and apply `pipeline.transform()` to the held-out rows — and explain why `setup()` alone is not a leak-free split.
- Extract JSON from a REST API with `httpx` (GET with query parameters, POST with a JSON body), check the status, and turn the result into a typed DataFrame.
- Split a pipeline into modules with a `main.py` orchestrator.
- Compare pre- and post-cleaning profiles to measure cleaning effectiveness.
- Write a try/except wrapper around a network call or pipeline entry point that fails with a clear message.

### Drill answers

1. ```python
   for q in (0.999, 0.995):
       cap = taxi_clean["fare_sgd"].quantile(q)
       above = taxi_clean.filter(pl.col("fare_sgd") > cap).height
       print(f"P{q * 100:.1f} = S${cap:.2f}: {above} rows above")
   ```
   The 99.9th percentile is S$33.72 with 44 rows above it; the 99.5th is S$17.41 with 218 rows above — 174 more. Neither cap is obviously right. Fares above S$17 are plausible for long trips (`distance_km` goes up to about 100 km in this log), so dropping them would bias a fare model against long journeys. Check `distance_km` for the high-fare rows first: a high fare on a long trip is real, a high fare on a 1 km trip is an error. Capping (`pl.col("fare_sgd").clip(upper_bound=cap)`) keeps the row but limits its influence.
2. ```python
   print(taxi_clean.group_by("time_period").agg(pl.col("km_from_cbd").mean().round(2)).sort("km_from_cbd"))
   ```
   Every period averages about 14.8 km (morning peak 14.75, late night 14.77, off-peak and evening peak 14.85). There is no commuter pattern: in this synthetic log, pickup location is independent of time of day. In a real taxi log you would expect morning-peak pickups in the suburbs and evening-peak pickups near the CBD — a feature that carries no signal here might carry a lot on real data.
3. ```python
   print([c for c in result.train_data.columns if c.startswith(("time_period", "is_weekend", "payment_type"))])
   ```
   You get `is_weekend_false`, `is_weekend_true`, four `time_period_*` columns (`evening_peak`, `late_night`, `morning_peak`, `off_peak`) and four `payment_type_*` columns (`Card`, `Cash`, `Grab`, `NETS`). `hour_of_day` was *not* one-hot encoded: it is an integer, so PreprocessingPipeline treated it as numeric and standardised it (mean 0, std 1). A linear model will therefore see the hour as a straight line — "later is more" — and cannot learn that both 8 AM and 6 PM are busy. If the hour matters as a category, cast it to a string before `setup()`, or rely on the `time_period` categories.
4. Run `r_mean = PreprocessingPipeline().setup(data=train_rows, target="fare_sgd", normalize=True, categorical_encoding="onehot", imputation_strategy="mean")`. The two `train_data` frames are identical (`r_mean.train_data.equals(result.train_data)` is `True`). By Step 5 the cleaning has already removed or filled every null in the feature columns, so there is nothing left to impute and the strategy makes no difference. Imputation choices only matter for nulls that reach the pipeline — check `model_df.null_count()` before you spend time tuning them.
5. One possible `extract.py`:
   ```python
   # extract.py
   import httpx
   import polars as pl

   from shared import MLFPDataLoader

   LOCATION_COLUMNS = {"place": pl.String, "lat": pl.Float64, "lng": pl.Float64}


   def load_taxi_data() -> pl.DataFrame:
       """Load the raw taxi log and parse its timestamp strings."""
       raw = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
       return raw.with_columns(
           pl.col("pickup_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
           pl.col("dropoff_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
       )


   def fetch_locations(search: str) -> pl.DataFrame:
       """Search OneMap; return an empty, correctly-typed frame if the call fails."""
       try:
           response = httpx.get(
               "https://www.onemap.gov.sg/api/common/elastic/search",
               params={"searchVal": search, "returnGeom": "Y", "getAddrDetails": "N", "pageNum": 1},
               timeout=10,
           )
           response.raise_for_status()
       except httpx.HTTPError as exc:
           print(f"OneMap request failed ({exc}); continuing without locations")
           return pl.DataFrame(schema=LOCATION_COLUMNS)
       results = response.json()["results"]
       if not results:
           return pl.DataFrame(schema=LOCATION_COLUMNS)
       return pl.DataFrame(results).select(
           pl.col("SEARCHVAL").alias("place"),
           pl.col("LATITUDE").cast(pl.Float64).alias("lat"),
           pl.col("LONGITUDE").cast(pl.Float64).alias("lng"),
       )
   ```
   `transform.py` holds the Step 3 and Step 4 code as two functions that take and return a DataFrame, and `main.py` follows the pattern in "Project structure". The test of a good split: you can call `clean_taxi_data(load_taxi_data())` from a notebook without running anything else, and a network failure in `fetch_locations` prints a clear message instead of crashing the pipeline.

---

# Chapter Summary

You started this chapter not knowing what a variable was. You are ending it having run a full end-to-end data pipeline on a messy Singapore-style dataset — and having found, in every course dataset, problems nobody told you were there. That is a non-trivial jump. Before moving to Module 2, take five minutes to consolidate the picture.

## The shape of what you learned

Module 1 has a clear internal arc. Lessons 1.1 through 1.3 are pure Python and Polars fundamentals — the alphabet of data work. Lessons 1.4 through 1.6 are the vocabulary: joins let you combine datasets, window functions let you express time-series patterns, visualisation lets you communicate what you have found. Lessons 1.7 and 1.8 are the grammar: Kailash engines that take your hand-built patterns and run them automatically, so you can apply them at scale without re-typing.

The through-line is the idea that you should never trust a number you did not produce yourself. Every lesson teaches you one more class of numbers you can produce and trust. At the start of the chapter you were at the mercy of whatever dashboard someone else had built. At the end you can write your own.

## The four Polars patterns to internalise

The four patterns below make up roughly 80% of the Polars code you will ever write. Keep them in mind; when a task maps naturally onto one of them, use it. When a task does not, pause and ask whether you are over-complicating things.

**Pattern 1: Filter-select-sort.** `df.filter(...).select(...).sort(...)`. Use for exploratory queries: "show me this subset of the data, these columns, in this order". Lesson 1.2.

**Pattern 2: Group-aggregate.** `df.group_by(col).agg(expressions)`. Use for summary tables: "one row per group with these statistics". Lessons 1.3 and 1.4.

**Pattern 3: With-columns (feature engineering).** `df.with_columns((expression).alias("name"))`. Use for derived values: "add a new column computed from existing ones". Lessons 1.2, 1.4, 1.5, 1.8.

**Pattern 4: Window-over.** `df.with_columns(pl.col("x").window_function().over("partition"))`. Use for per-row contextual features: rolling means, YoY changes, ranks within group. Lesson 1.5.

Every complex pipeline you build will combine these four patterns. You do not need more patterns for Module 1 or most of Module 2.

## The three Kailash engines and their contracts

**DataExplorer.** Profile a DataFrame, surface quality issues as alerts, compare two DataFrames for drift, generate HTML reports. Input: a DataFrame. Output: a profile object. Use at the beginning and end of every pipeline.

**PreprocessingPipeline.** Impute, scale and encode a DataFrame for model training, and re-apply the same learned rules to new rows. Input: a DataFrame with a designated target column. Output: a result object with train_data and test_data, plus `transform()` for new rows. `setup()` fits on every row it is given before it splits — so hold out your test rows first, call `setup()` on the training rows, and `transform()` the rest. Use at the end of cleaning, just before training.

**ModelVisualizer.** Build interactive charts (histogram, scatter, box plot, bar via `metric_comparison`, line via `training_history`) from Polars DataFrames, dropping to Plotly for heatmaps and stacked bars. Input: a DataFrame and chart configuration. Output: a Plotly Figure — check its axis labels when you repurpose a method. Use throughout the pipeline for exploration and reporting.

These three engines cover 90% of your data-pipeline needs in Modules 1 and 2. The other 10% you will handle in pure Polars, which is fine — Polars and the engines are designed to play together.

## What Module 2 builds on

Module 2 is "Statistical Mastery for Machine Learning and Artificial Intelligence (AI) Success". It assumes:

- You can write Polars filters, aggregations, and window functions without looking things up.
- You understand the difference between mean, median, variance, and standard deviation, and can explain when each is appropriate.
- You can profile a DataFrame and interpret the alerts.
- You can hold out test rows and prepare features with PreprocessingPipeline without leaking test information.
- You can make an interactive chart of anything and export it as HTML.

If any of these feels uncertain, spend an hour on the corresponding lesson's "Try It Yourself" drills before moving on. Module 2 will not slow down to re-teach.

Module 2 introduces:

- **FeatureStore** — a versioned store for feature groups, built on the same join patterns you learned in Lesson 1.4.
- **FeatureEngineer** — automated feature generation: interactions, polynomial features, lag features, target encoding. All of these build on the `with_columns` pattern you learned in Lesson 1.5.
- **ExperimentTracker** — a log of your runs, parameters, and metrics, for reproducibility and comparison.
- **Inferential statistics.** Regression, logistic regression, ANOVA, hypothesis testing, power analysis, Bayesian updating. The statistical foundations that Module 3 will build a full ML pipeline on.

You will meet MLE, Fisher information, Bayesian priors, and hypothesis testing formally in Module 2. The intuitive versions you met in this chapter's THEORY sections are warm-up; Module 2 will do the derivations.

## What you should do before moving on

Three things:

1. **Re-run the worked examples** from at least Lessons 1.3, 1.5, and 1.8. Type them out, do not copy-paste. The muscle memory matters.
2. **Run the end-to-end pipeline on a dataset of your own choice.** Any real Singapore dataset from `data.gov.sg` will do — and on real data, check whether the trends and seasonality that the synthetic course files lacked are actually there. Load it, profile it, clean it, visualise it, generate a report. Fifteen minutes, and you will solidify the pattern.
3. **Tell someone what you learned.** Teaching is the best test of understanding. Pick a non-technical friend and explain the "dashboard that said everything was fine" scenario, why the median is preferable to the mean for property prices, and what a histogram reveals that a summary statistic cannot. If you can explain it, you know it.

Then take a day off. Come back to Module 2 rested. You will need it — Module 2 is longer and more formal than Module 1, and the payoff is cumulative.

---

# Glossary

Every technical term introduced in this chapter, defined plainly.

**Aggregation.** The process of collapsing many rows into a single summary value, usually within groups. Mean, median, count, sum, and standard deviation are aggregations. See `group_by` and `agg`.

**Alert.** A structured warning from DataExplorer indicating that a column or dataset crosses a configurable quality threshold. Each alert is a dict with a `type` (one of eight, like `high_skewness`), a `severity` (`info` or `warning`), a `column` (or `columns` for a correlated pair), and a `value`.

**AlertConfig.** The configuration object for DataExplorer that controls which thresholds trigger alerts. Tuning AlertConfig is domain-specific work: defaults are not appropriate for every dataset.

**Async / await.** Python keywords for asynchronous functions. An `async def` function returns a coroutine; `await` pauses execution until the coroutine completes. DataExplorer's `profile`, `compare`, and `to_html` methods are async.

**Calendar spine.** A complete table of every period (for example, every town × every month) that observed data is left-joined onto, so gaps become explicit nulls before window functions run.

**Bar chart.** A chart showing one bar per category, with bar height proportional to a value. Best for comparing a metric across categories.

**Bimodal distribution.** A distribution with two distinct peaks. Indicates two sub-populations mixed together — for example, genuine records mixed with a batch of mis-recorded ones.

**Boolean.** A value that is either `True` or `False`. The result of a comparison like `price > 500_000`. Python's `bool` type.

**Cardinality.** The number of unique values in a column. High-cardinality columns (ratio near 1) often cannot be one-hot encoded and need binning or target encoding.

**Categorical.** A column whose values are discrete categories (town names, flat types) rather than continuous numbers. Encoded as one-hot or ordinal for ML.

**Coefficient of variation (CV).** The standard deviation divided by the mean, often expressed as a percentage. A scale-invariant measure of spread.

**Column.** One dimension of a DataFrame. A named sequence of values all of the same type.

**Correlation.** A number between -1 and +1 measuring the strength and direction of a relationship between two variables. Pearson measures linear relationships; Spearman measures monotonic ones.

**DataExplorer.** The Kailash ML engine for automated dataset profiling.

**DataFrame.** A two-dimensional rectangular table of data with named columns. The fundamental data structure in this course.

**Describe.** A Polars method that computes basic statistics (count, null count, mean, std, min, max, quartiles) for every column in one call.

**Duplicate.** A row that is exactly identical to another row in the dataset. DataExplorer flags a dataset with more than the configured threshold of duplicates.

**ETL.** Extract-Transform-Load. The three-stage pattern for every data pipeline: get data, clean it, write it out.

**Expression (Polars).** A description of a computation on a column, not the computed result. Created with `pl.col("name")` and chained with methods. Evaluated only when passed to `.filter`, `.with_columns`, `.agg`, etc.

**f-string.** A Python string literal prefixed with `f` that supports variable interpolation and formatting inside curly braces. `f"Price: S${price:,.0f}"`.

**Feature engineering.** The process of creating new columns from existing ones to help a downstream ML model. Extracting hour-of-day from a timestamp is feature engineering.

**Filter.** Keep only the rows of a DataFrame where a given Boolean condition is true. `df.filter(pl.col("price") > 500_000)`.

**Forward fill.** Replacing a NULL with the most recent non-NULL value in a time-ordered series. Standard for upsampling quarterly data to monthly.

**Function.** A named, reusable block of code that takes parameters and returns a value. Defined with `def`.

**Gestalt principles.** Rules about how the human visual system groups visual elements. Proximity, similarity, closure, continuity, connection, enclosure.

**Group-by.** Splitting a DataFrame into groups based on one or more key columns, then aggregating each group separately. The SQL `GROUP BY`.

**Haversine distance.** The great-circle distance between two points on a sphere, computed with the haversine formula. Used for geospatial feature engineering.

**Heatmap.** A grid of coloured cells where colour encodes a numeric value. Used for correlation matrices and confusion matrices.

**Histogram.** A chart showing the distribution of a numeric column by binning values and counting per bin.

**Imputation.** Filling in missing values with a substitute (median, mean, mode, or model-based estimate) so the data can be used by downstream models.

**Inner join.** A join that keeps only rows where the key exists in both tables. Non-matching rows are dropped from both sides.

**Join.** An operation that combines two tables by matching rows on a shared key. Inner, left, right, and outer (Polars: `full`) are the four types. A join multiplies rows when the key is not unique on the right side.

**JSON.** JavaScript Object Notation — the text format most web APIs return: nested objects and lists that map onto Python dicts and lists.

**Lazy frame.** A Polars query plan that is not executed until `.collect()` is called. Enables query optimisation (predicate pushdown, projection pushdown).

**Left join.** A join that keeps all rows from the left table, filling right-side columns with NULL where no match is found. The most common join type in practice.

**Line chart.** A chart connecting data points with line segments in x-axis order. Standard for time-series data.

**Mean.** The arithmetic average: sum of values divided by count. Sensitive to outliers.

**Median.** The middle value when the data is sorted. Robust to outliers. Preferred for skewed distributions like income and prices.

**Method chaining.** Calling multiple methods on a DataFrame in sequence, where each method's output is the next method's input. The idiomatic style of Polars code.

**Mode.** The most frequently occurring value. Appropriate for categorical data, less useful for continuous data.

**ModelVisualizer.** The Kailash ML engine for producing interactive Plotly charts (histograms, scatter plots, box plots, bar charts, line charts) from Polars DataFrames. Heatmaps and stacked bars are built with Plotly directly.

**Null.** A typed marker for "missing value". Different from zero, empty string, or NaN. Polars has first-class null support.

**Outlier.** A value far from the bulk of the distribution. Sometimes real, sometimes an error. Always worth investigating before including in aggregations.

**Parquet.** A columnar file format that stores typed data efficiently. Much faster than CSV for large numeric datasets.

**Polars.** The DataFrame library used throughout this course. Fast, memory-efficient, polars-native (no pandas bridge), written in Rust.

**PreprocessingPipeline.** The Kailash ML engine that imputes, scales, encodes, and splits a DataFrame for model training. Its `setup()` fits on all rows passed in, so hold out test rows first and apply `transform()` to them.

**Quantile.** A percentile of a distribution. The 25th quantile (Q1) is the value below which 25% of the data falls. The median is the 50th quantile.

**REST API.** A web service you query with HTTP requests — GET to retrieve data (with query parameters), POST to send it — usually returning JSON.

**Rank.** A column that assigns each row a position within its partition. `rank(method="ordinal", descending=True)` gives 1 to the highest value, 2 to the next, and so on.

**Right skewed.** A distribution with a long tail on the right. Most values are small with a few very large ones. Typical of prices, incomes, populations. Mean > median.

**Rolling mean.** A moving average: each value is replaced with the mean of itself and the previous $k-1$ values. Smooths time-series noise.

**Scatter plot.** A chart with one point per observation, one variable on each axis. Reveals relationships, outliers, and clusters.

**Schema.** The list of column names and their types. Use `df.columns` and `df.dtypes` in Polars.

**Select.** Keep only the named columns of a DataFrame, dropping the rest. `df.select("col1", "col2")`.

**Shift.** Move values in a column forward or backward by a fixed number of positions, with NULLs filling the gap. Combined with `.over(partition)` for within-group shifts.

**Skewness.** A measure of asymmetry in a distribution. Zero for symmetric, positive for right-skewed, negative for left-skewed. DataExplorer flags columns with high absolute skewness.

**Sort.** Order the rows of a DataFrame by one or more columns, ascending or descending.

**Spearman correlation.** A rank-based correlation that captures monotonic (not necessarily linear) relationships.

**Standard deviation.** The square root of the variance. A measure of spread in the original units.

**Standardisation.** Rescaling a numeric column to have mean 0 and standard deviation 1. Done by PreprocessingPipeline when `normalize=True`.

**String.** A sequence of text characters, written inside quotes in Python. Polars type is `pl.String` (alias for `pl.Utf8`).

**Summary statistic.** A single-number summary of a column: mean, median, std, min, max, quantile.

**Tuple.** An ordered, immutable sequence of values in Python. `df.shape` returns a tuple like `(rows, cols)`.

**Type.** The kind of value a variable or column holds. Python types include `int`, `float`, `str`, `bool`. Polars column types include `Int64`, `Float64`, `String`, `Boolean`, `Date`, `Datetime`.

**Variance.** The mean of the squared deviations from the mean. In squared units; for original units use the standard deviation.

**Window function.** A computation that produces a value for each row based on a set of related rows, without collapsing the DataFrame. Rolling means, YoY changes, and ranks are window functions. See `.over()`.

**YoY (year-over-year).** The percentage change between a value and the same value from exactly one year earlier. Comparing like months cancels out a seasonal pattern. Needs a gap-free calendar spine if computed with `shift(12)`.

---

# Further Reading

The following are standard references that expand on material covered in this chapter. You do not need them to complete Module 1, but they are the books and papers practitioners refer to when they want to go deeper.

**On data work in general**

- Wickham, Hadley, and Garrett Grolemund. *R for Data Science.* O'Reilly, 2017 (second edition 2023). The canonical beginner-to-intermediate reference for tidy data work. Uses R and the tidyverse, not Python and Polars, but the concepts translate directly. The chapters on "Explore" and "Wrangle" are the complement to what you just learned. Free online at `r4ds.hadley.nz`.

- McKinney, Wes. *Python for Data Analysis.* O'Reilly, 2012 (third edition 2022). The pandas reference, written by pandas' author. We do not use pandas in this course, but the conceptual material on aggregation, joins, and reshaping is the same. The chapter on data aggregation and group operations is particularly relevant to Lessons 1.3 and 1.5.

**On Polars specifically**

- Polars documentation, `pola.rs`. The official reference, well-maintained and increasingly comprehensive. The "User Guide" sections on expressions, lazy evaluation, and window functions are excellent.

- Janssens, Jeroen, and Thijs Nieuwdorp. *Python Polars: The Definitive Guide.* O'Reilly, 2025. A book-length treatment of Polars: expressions, lazy evaluation, performance, and advanced patterns beyond what this textbook touches.

**On visualisation**

- Tufte, Edward. *The Visual Display of Quantitative Information.* Graphics Press, 1983 (second edition 2001). The foundational book on chart design, and the source of most of the principles in Lesson 1.6. The chapter on "chartjunk" and the "lie factor" are essential reading for anyone producing charts for others.

- Cleveland, William. *The Elements of Graphing Data.* Hobart Press, 1985. The empirical complement to Tufte: what the human visual system can and cannot judge accurately (position along a common scale beats length, angle and area — the reason Lesson 1.6 prefers bars to pies).

- Wilke, Claus. *Fundamentals of Data Visualization.* O'Reilly, 2019. Modern, well-illustrated, and free online at `clauswilke.com/dataviz/`. The chapters on "Common pitfalls of color use" and "Handling overlapping points" are directly applicable to Lesson 1.6's scatter-plot work.

- Plotly documentation, `plotly.com/python`. The reference for Plotly, which is what ModelVisualizer uses under the hood. Most ModelVisualizer customisations are just Plotly method calls on the returned Figure.

**On statistics (background for Module 2)**

- Wasserman, Larry. *All of Statistics.* Springer, 2004. A compact, technically rigorous introduction to modern statistics for people with a mathematics background. Covers probability, estimation, hypothesis testing, Bayesian inference, and bootstrap — all in about 400 pages. Will be useful in Module 2.

- Efron, Bradley, and Trevor Hastie. *Computer Age Statistical Inference.* Cambridge, 2016. A history of statistics from classical methods to modern machine learning, with working code examples. The chapter on the bootstrap is particularly elegant. Free online from the authors' website.

- Anscombe, Francis. "Graphs in Statistical Analysis." *The American Statistician*, 1973. The original Anscombe's quartet paper. Four pages, and worth reading in full.

**On data quality and profiling**

- Redman, Thomas. *Data Driven: Profiting from Your Most Important Business Asset.* Harvard Business Review Press, 2008. The business case for data quality, written for managers — useful background for the motivation of Lesson 1.7.

**On Singapore-specific data sources**

- `data.gov.sg` — the Singapore government's open data portal. HDB resale prices, economic indicators, weather and many more Singapore datasets are published here for free download. The Module 1 course datasets are synthetic: they are modelled on the structure of public datasets like these, but their values are generated for teaching (with data-quality problems planted on purpose), so do not quote numbers from them as facts about Singapore. For real figures, go to the source.

- OneMap API documentation (linked from `onemap.gov.sg`) — the service used for the REST extraction examples in Lesson 1.8. Provides search/geocoding, routing, and map services for Singapore; some endpoints require a free account token. When you do Drill 5 of Lesson 1.8, OneMap is often the fastest way to enrich addresses with coordinates.

- Monetary Authority of Singapore (MAS) statistics portal. Official exchange rates, monetary aggregates, and financial stability indicators — where you would get real FX data to replace the synthetic series used in Lesson 1.7.

**Papers on the specific topics this chapter skimmed**

- Tukey, John. *Exploratory Data Analysis.* Addison-Wesley, 1977. The book that named and popularised the field. Tukey's stem-and-leaf plots are out of fashion but his philosophy — "the greatest value of a picture is when it forces us to notice what we never expected to see" — is the animating principle of this entire chapter.

- Huber, Peter. *Robust Statistics.* Wiley, 1981 (second edition with Ronchetti 2009). The rigorous reference for robust statistics. If you want to know why the MAD is multiplied by 1.4826, this is where the derivation lives.

- Pearson, Karl. "Notes on the History of Correlation." *Biometrika*, 1920. The history of the correlation coefficient, written by its co-inventor. Short, readable, and useful context.

---

*You have finished the reading for Module 1. The exercises in `modules/mlfp01/` are next. Good luck.*
