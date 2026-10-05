# Module 1 (MLFP01): Machine Learning Data Pipelines and Visualisation Mastery with Python — Speaker Notes

Master speaker notes for the module deck (`deck.html`, 78 slides). One section per
slide, in deck order, numbered and titled exactly as the deck. Per-lesson pages with
deeper prose live in `lessons/NN/notes.html`; these master notes are the
slide-by-slide companion for the instructor presenting the full deck.

Total teaching time: ~196 minutes (about 3 hours 15 minutes), plus hands-on exercise
time (~60-90 minutes across the 8 exercises) and breaks. Plan for a full day, or two
half-day sessions split after Lesson 1.4 (Slide 42).

How to read each section:

- **Hook** — say this first, before advancing into the slide's detail.
- **Key question** — pose it to the room and wait for answers; do not answer it yourself.
- **Beginner cue** — what to do if the room looks lost.
- **Advanced cue** — what to offer experienced learners who look bored.
- **Transition** — the line that hands off to the next slide.

The course HDB and taxi datasets are synthetic and are described to students as
synthetic. Never present them as official records.

---

## Slide 1: Data Pipelines and Visualisation

**Time:** ~2 min

**Hook:** Read the question on screen aloud, slowly: "Can you trust a number you
didn't explore yourself?" Let it sit for five seconds before speaking again.

Welcome the room to Module 1 of the ML Foundations for Professionals course at
Terrene Open Academy. This module takes learners from zero Python to building
complete data pipelines in a single day. Make the promise concrete: by the end of
today, every person in the room will load roughly 50,000 HDB resale transactions
and find problems in them that a summary dashboard hides completely.

Set expectations for the mixed audience immediately. Reassure beginners that no
prior programming is needed — the module starts from the very first line of code.
For experienced practitioners, name the destinations so they know the day is not
a waste: window functions, lazy frames, and automated data profiling with Kailash
engines by the afternoon.

**Beginner cue:** If anyone looks anxious, say: "Every line of code today is
typed together. You can copy every character and still succeed."

**Advanced cue:** "The Foundations layer will feel slow for the first hour. The
Theory and Advanced slides carry the depth — lazy evaluation internals, profiling
algorithms, and pipeline design."

**Transition:** "Let me show you what you will be able to do after these 8 lessons."

---

## Slide 2: What You Will Learn

**Time:** ~2 min · Foundations

**Hook:** "Here is the contract for today — five things you will be able to do by
five o'clock."

Walk through the outcomes column. These are concrete, checkable skills — loading
data, filtering it, aggregating it, joining tables, charting it, and profiling it
automatically — not abstract topics. Read two or three aloud and ask learners to
read the rest silently.

Then explain the three-layer system, because it governs how each person should
spend their attention. Green Foundations slides are for everyone — they are the
spine of the day. Blue Theory slides add mathematical depth for those who want it.
Purple Advanced slides are expert tangents and can be skipped without penalty.

**Beginner cue:** "Green is all you need. If you follow every green slide today,
you pass every exercise."

**Advanced cue:** Point at the Theory and Advanced markers: "Lazy evaluation
internals and profiling algorithms live on the blue and purple slides. Do not
switch off during the green ones — the idioms matter."

**Key question:** "Which of these five outcomes is the reason you are here?" A
quick show of hands tells you the room's motivation mix.

**Transition:** "Here is the roadmap for the 8 lessons."

---

## Slide 3: Your Journey: 8 Lessons

**Time:** ~2 min

**Hook:** "Eight lessons, one day, one dataset at a time."

Walk the table one sentence per lesson — do not read every cell. Emphasise the
arc: Lessons 1.1 to 1.4 teach Python through data tasks, so programming and data
skills arrive together. Lesson 1.5 adds time-series depth with window functions.
Lesson 1.6 makes numbers visible. Lessons 1.7 and 1.8 bring in the Kailash engines
and end with a capstone that integrates everything into a real pipeline.

Reassure the room that each lesson builds directly on the previous one — there are
no jumps. Every exercise uses data they have already met in the slides.

**Beginner cue:** "If you are still with me after Lesson 1.1, you can finish the
day. The slope is gentlest at the start by design."

**Advanced cue:** "The capstone in 1.8 is where it comes together — extract,
profile, clean, prepare, visualise, report. Judge the module by that pipeline."

**Transition:** "Before we start coding, let me introduce the three Kailash
engines you will use."

---

## Slide 4: Kailash Engines You Will Meet

**Time:** ~2 min

**Hook:** "Three power tools. You will learn the hand-tool version of each first."

Introduce the three engines in one breath each: **DataExplorer** profiles a whole
dataset in one call — types, distributions, correlations, data-quality alerts.
**PreprocessingPipeline** turns a raw table into model-ready data — imputation,
encoding, scaling, train/test split. **ModelVisualizer** produces the standard
ML charts with sensible defaults.

Explain the pedagogical order, because it is deliberate: Lessons 1.1 to 1.5 teach
core Python and Polars so learners understand what the engines automate. Engines
appear from Lesson 1.6 onward. When an engine produces a surprising result, a
learner who did it manually first can debug it; one who did not, cannot.

**Beginner cue:** "Think of hand tools versus power tools. We spend the morning on
hand tools so the power tools are never mysterious."

**Advanced cue:** "DataExplorer's methods are async. Module 1 uses small sync
wrappers — `shared.run_profile` and friends — so you can use the engines before
async is taught in a later module."

**Transition:** "Let us make sure everyone's environment works, then we begin."

---

## Slide 5: Tools and Setup

**Time:** ~2 min

**Hook:** "Two ways to run every exercise today. Pick one in the next two minutes."

Present the two exercise formats. Local Python files for VS Code live in
`modules/mlfp01/local/` — best for learners who want the developer experience and
version control. Self-contained Colab notebooks live in
`modules/mlfp01/colab-selfcontained/` — nothing to install, no git clone; the
helpers are inlined and the data downloads automatically. Both formats produce
identical results and identical checkpoints.

For the local format, `uv sync` installs the environment in seconds; `pip` works
too. For Colab, learners simply open the notebook link. Check that everyone can
run the first cell before moving on — this is the last setup moment of the day.

**Beginner cue:** "If you are not sure, use Colab. Zero install, and the notebook
carries everything it needs."

**Advanced cue:** "Local .py files are what production code looks like. If you
already run a Python setup, use them."

**Transition:** "Everything is set up. Let me tell you why today matters."

---

## Slide 6: The Story That Starts Everything

**Time:** ~3 min · Foundations

**Hook:** "A dashboard that said everything was fine — while the raw data screamed."

Say this explicitly before anything else: this is an **illustrative scenario, not
a reported incident**. The course's HDB file is a synthetic dataset modelled on
the public HDB resale data, with data-quality problems planted on purpose so
learners can practise finding them. Never present the story as a real event, and
never attribute the data to an agency.

Tell the story: an analytics team publishes a monthly median resale-price
dashboard. Hidden in the raw transactions is a batch of records with impossible
prices — data-entry errors, mis-keyed figures. The median barely moves, so the
dashboard shows nothing unusual — and every model trained on that data quietly
learns from the errors. The moral: a median is robust to extremes _by design_,
which is exactly why it hides them.

**Beginner cue:** "A dashboard is like a weather report — it tells you the average
temperature, not that it hailed on your street."

**Advanced cue:** Ask: "What would a _mean_-based dashboard have shown instead?"
Answer: the bad prices pull the mean visibly, while the median does not move.
Robustness is a double-edged sword.

**Key question:** "Who here has shipped a number they did not check against the
raw rows?" Let a few hands go up — it normalises the lesson.

**Transition:** "What does a summary hide? Let us look at exactly what went wrong."

---

## Slide 7: What Went Wrong?

**Time:** ~3 min · Foundations

**Hook:** "Three design choices, one blind dashboard."

Walk the three failure modes on the left: aggregation hides outliers — a median
over 50,000 transactions does not move for 251 bad ones; pre-built charts answer
yesterday's questions and cannot ask new ones; and a single number is not a
distribution — without a histogram, the impossible values are invisible.

Then the punchline, on the course's own data: the minimum is S$10 (107 rows — a
flat for the price of two coffees) and the maximum is S$9,000,000 (144 rows),
while the median sits at about S$849k and never noticed. Point at the three-line
code preview and promise: "By the end of today you will write these three lines
yourself — `min`, `median`, `max` — and that is all it takes to catch this."

**Beginner cue:** The takeaway sentence is: "Always look at the raw data yourself
before trusting a summary."

**Advanced cue:** "Automated profiling in Lesson 1.7 catches exactly this —
DataExplorer reports skewness and per-column outlier counts. Today you learn to
see it by hand first."

**Transition:** "This is why you learn exploratory data analysis. Let us start
with Lesson 1.1."

---

## Slide 8: Lesson 1.1: Your First Data Exploration

**Time:** ~2 min · Foundations

**Hook:** "Forty-five minutes from now, you will have loaded real data and asked it
three questions."

This is the Lesson 1.1 overview slide. Frame the two halves: pure Python first
(variables, arithmetic, strings), then data exploration with Polars. Neither half
works without the other — learners need variables to hold values before they can
hold a dataset.

**Beginner cue:** "We start from the very basics. Nothing today assumes you have
seen code before."

**Advanced cue:** "Skim the Python basics at your own pace; the Polars idioms in
the second half are what the rest of the course builds on."

**Transition:** "Let us start with the most fundamental building block: variables."

---

## Slide 9: Variables and Assignment

**Time:** ~3 min · Foundations

**Hook:** "A variable is a name taped onto a value. That is the whole idea."

Variables are labels attached to values; Python figures out the type automatically.
Walk the examples on screen one at a time — a string for the town, an integer for
the price, a float for the area. Read each assignment as an English sentence:
"town gets the value Queenstown."

F-strings are the modern way to embed variables in text, and the `f` before the
quote is required — without it the braces print literally. The `:,` format
specifier adds thousand separators, which matters the moment prices like 850000
need to be read as 850,000.

**Beginner cue:** Compare a variable to a spreadsheet cell that has a name instead
of an address like A1. Names are easier to think about than addresses.

**Advanced cue:** Mention dynamic typing and `type()` for introspection — and that
unlike spreadsheets, a name can be re-bound to a different type, which is a common
source of bugs.

**Key question:** "What prints if I forget the `f` in front of the string?" Run it
live and let them see the literal braces.

**Transition:** "Now let us do arithmetic with these variables."

---

## Slide 10: Arithmetic and Expressions

**Time:** ~3 min · Foundations

**Hook:** "Every operator on this slide answers a real data question."

Anchor each operator to a use case: division for ratios like price per square
metre, modulo for binning and parity checks, exponentiation for compound growth.
Walk the `price_per_sqm` calculation step by step — price divided by floor area —
because it is the first computed column learners will derive in the exercises.

Show the f-string format specifier `:,.0f` — comma-separated, zero decimal places —
as the standard way to print money. Students will reuse this pattern in every
exercise reflection.

**Beginner cue:** Walk the price-per-sqm line slowly: numerator, denominator,
unit. "Dollars per square metre — the number property agents actually quote."

**Advanced cue:** Operator precedence follows standard BODMAS; when in doubt, use
parentheses — they cost nothing and remove ambiguity for the reader.

**Transition:** "Now that we can compute values, let us format and display them
properly."

---

## Slide 11: Strings and F-Strings

**Time:** ~3 min · Foundations

**Hook:** "Half the data you will ever touch is text — town names, flat types,
column names."

Strings are text data, and in data analysis they are everywhere: column names,
categories, labels, messages. Demonstrate the basic f-string pattern first —
`f"Town: {town}"` — and only then the format specifiers. Any expression can go
inside the braces, not just a variable name; show `f"{price / area:,.0f}"` to
make the point.

**Beginner cue:** Focus on the basic pattern. Formatting details come with
practice — nobody memorises specifiers on day one.

**Advanced cue:** F-strings replaced `.format()` and `%` formatting in modern
Python; learners maintaining old code will meet all three.

**Key question:** "Why does `f"{850000:,}"` print `850,000` — and what would print
without the colon?" Let them predict before you run it.

**Transition:** "We have been working with single values. Data analysis needs
collections of values. Enter Polars."

---

## Slide 12: Your First DataFrame

**Time:** ~3 min · Foundations

**Hook:** "One line to load a dataset. That is the moment today becomes real."

`import polars as pl` is the community convention, exactly like `import numpy as np`
— learners will see it in every example they ever read. The `MLFPDataLoader` finds
the course file for you, locally or in Colab; under the hood it calls
`pl.read_csv` and returns a DataFrame, the central data structure for the entire
course.

Describe the weather file honestly: 12 rows of monthly climate averages prepared
for this course — illustrative values, not official station records. Then the
three inspection moves: `shape` gives (rows, columns), `columns` lists the names,
`head()` shows the first five rows. Run them live.

**Beginner cue:** "A DataFrame is a spreadsheet where every column has a name and
a type — and the types are enforced."

**Advanced cue:** Polars uses the Apache Arrow memory format, which is why it is
faster than pandas and why interop with other Arrow-based tools is zero-copy.

**Transition:** "Let us dig deeper into what the data actually contains."

---

## Slide 13: Inspecting Your Data

**Time:** ~3 min · Foundations

**Hook:** "If you memorise one function today, memorise this one."

`describe()` is the single most important function in exploratory data analysis —
always run it first on new data. Read the weather output together: temperature
barely moves (mean 27.47, std 0.62 — a coefficient of variation around 2%),
rainfall moves a lot (mean 171.75, std 40.10 — CV around 23%). Singapore is hot
with variable rain; the statistics say so in two lines.

Warn about the string-column trap explicitly: for the `month` column, `describe()`'s
min and max are *alphabetical* — April and September — and say nothing about
temperature. The hottest month is May at 28.3 °C. Students who read min/max as
"coolest/hottest month" have made their first analytical error.

**Beginner cue:** Focus on count, mean, min, max. Standard deviation can wait —
say it is "how far values typically sit from the average" and move on.

**Advanced cue:** Polars `describe()` also reports the 25th, 50th and 75th
percentiles — the skeleton of the distribution.

**Key question:** "The max row says September for month — does that mean September
is hottest? Why not?"

**Transition:** "Let us see how Polars tells us about column types."

---

## Slide 14: Data Types in Polars

**Time:** ~2 min · Foundations

**Hook:** "Polars refuses to guess. That strictness is a feature, not friction."

Polars is strict about types, unlike pandas which silently coerces — a column of
numbers with one stray string stays typed, and the error surfaces loudly instead
of corrupting a later calculation. The `schema` property is the quickest way to
see every column's type at once; read the weather schema together.

**Beginner cue:** "Types tell the computer how to store and compute on data.
Numbers can be averaged; strings cannot. That is the whole story for today."

**Advanced cue:** Internally these are Arrow types — Int64, Float64, String,
Boolean — which is what enables zero-copy exchange with other Arrow tools.

**Transition:** "Let us put it all together with the exercise."

---

## Slide 15: Exercise 1.1: Singapore Weather Exploration

**Time:** ~3 min (exercise work time ~10 min) · Foundations

**Hook:** "Your first real analysis. Twelve rows, three questions, five minutes."

Walk through the exercise requirements. The shared data loader handles file paths,
so nobody fights with absolute paths. Emphasise the assessment rule: answers must
cite specific numbers from `describe()` output, not impressions.

Give the answers so you can check the room: 12 months of data; highest monthly
mean temperature 28.3 °C; average monthly rainfall 171.75 mm. Note the deliberate
twist — `describe()` shows the hottest *value* but not *which month* it was;
finding that it is May is Task 8, and it requires the filtering skill that Lesson
1.2 formalises.

**Beginner cue:** Demonstrate the load-and-describe sequence live, step by step,
before releasing them.

**Advanced cue:** Challenge early finishers to compute rainfall's coefficient of
variation by hand (40.10 / 171.75 ≈ 23%) and compare it with temperature's ≈ 2%.

**Transition:** "You can now load data and inspect it. Next, we learn to filter
and transform it."

---

## Slide 16: Lesson 1.1 Recap

**Time:** ~1 min · Foundations

**Hook:** "Sixteen slides, three ideas."

Compress the lesson to its skeleton: variables hold values; Polars DataFrames
hold tables; `describe()` plus `shape`, `columns` and `head()` is the 30-second
ritual for any new dataset. This slide doubles as a reference card — tell learners
to photograph it.

**Beginner cue:** The Polars column on this recap is the most important takeaway
of the lesson.

**Advanced cue:** Natural break point — this is a good moment for a five-minute
stretch before filtering.

**Transition:** "In Lesson 1.2, we learn to ask questions of our data by filtering
and transforming it."

---
