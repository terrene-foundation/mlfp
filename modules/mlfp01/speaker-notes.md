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

## Slide 17: Lesson 1.2: Filtering and Transforming Data

**Time:** ~2 min · Foundations

**Hook:** "Every data question begins the same way: show me only the rows where…"

Filtering is how you ask questions of data — "show me only Queenstown flats under
S$700k". Frame the lesson: booleans, filters, column selection, computed columns,
and chaining them together. Note the forward reference on the slide: students will
see True/False values now, but Python `if/else` control flow is deliberately
deferred to Lesson 1.4 — say so, so nobody thinks it was forgotten.

**Beginner cue:** Compare filtering to a spreadsheet's filter dropdown — same
idea, but the condition is code, which means it is repeatable and shareable.

**Advanced cue:** Polars expressions compile to a query plan rather than iterating
row by row — the idiom is declarative: describe what you want, not how to loop.

**Transition:** "Let us start with boolean logic, the foundation of all filtering."

---

## Slide 18: Booleans and Comparisons

**Time:** ~3 min · Foundations

**Hook:** "The number one beginner bug in all of programming is on this slide."

`==` checks equality; `=` assigns a value. Say it twice and demonstrate the
mistake live — assign when you meant to compare — because every learner will make
it within the week. In Polars expressions, conditions combine with `&` and `|`,
not Python's `and` and `or`; the reasons (operator overloading on expressions)
can stay behind the curtain for now.

**Beginner cue:** "True and False are just yes/no answers to questions about
data. `price > 500000` asks a question; the answer is a column of yeses and nos."

**Advanced cue:** Mention short-circuit evaluation and that parentheses control
precedence — `&` binds tighter than comparison operators in Python, so the
parentheses on the slide are load-bearing, not style.

**Key question:** "Why does `a == 1 | b == 2` raise, but `(a == 1) | (b == 2)`
works?" Let the parentheses discussion happen before revealing.

**Transition:** "Now let us apply these comparisons to filter entire DataFrames."

---

## Slide 19: Filtering Rows with filter()

**Time:** ~3 min · Foundations

**Hook:** "One line of Polars replaces a thousand spreadsheet clicks."

`pl.col("town")` builds an expression referring to the town column; comparing it
with `==` produces a boolean mask; `df.filter(mask)` keeps the rows where the mask
is True. Read the code on screen as English: "filter where the town column equals
Queenstown and the price is below 700 thousand."

Two details to call out: parentheses around each condition are required when
combining with `&` or `|` — omit them and Python's precedence rules bite. And
`700_000` uses an underscore as a visual separator; Python ignores it, so use it
freely for money.

**Beginner cue:** Read every filter aloud as a sentence before running it. If the
sentence is wrong, the code is wrong.

**Advanced cue:** In lazy mode Polars pushes filter predicates into the scan, so
unmatched rows may never be read from disk — filtering early is also a
performance habit.

**Transition:** "Filtering selects rows. Let us also select specific columns."

---

## Slide 20: Selecting Columns and Sorting

**Time:** ~3 min · Foundations

**Hook:** "Choose your columns, choose your order — that is half of reporting."

`select()` picks columns; `filter()` picks rows. Those two verbs answer most
everyday data questions. Columns can be named as plain strings or as `pl.col()`
expressions — expressions let you transform on the fly, which previews
`with_columns` on the next slide.

Sorting by multiple keys reads left to right: first by town alphabetically, then
by price within each town. Demonstrate `descending=True` and ask what happens to
ties — a natural segue to deterministic ordering.

**Beginner cue:** "`select` is which columns do I want to see; `sort` is in what
order. Two separate questions, two separate methods."

**Advanced cue:** `select` accepts regex selectors like `pl.col("^price.*$")` —
handy on wide tables later in the course.

**Transition:** "Now the power move: creating new columns from existing ones."

---

## Slide 21: Creating Computed Columns

**Time:** ~3 min · Foundations

**Hook:** "Every feature you will ever engineer starts as a `with_columns` call."

`with_columns` is the most important transformation method in the course: it adds
new columns (or overwrites existing ones) from expressions. Walk the three
examples. Price per square metre — division, the canonical derived metric.
Remaining lease parsed from text — `remaining_lease` arrives as strings in two
formats ("71 years 11 months" and plain "92"), so `str.extract` pulls the leading
number and `cast` makes it an integer; about 1,474 rows have no remaining lease
and correctly stay null. A high-value flag — a boolean column from a comparison.

`.alias()` names the result; without it Polars names the column from the
expression, which gets unreadable fast.

**Beginner cue:** "It is the spreadsheet formula column, except the formula is
code and applies to 50,000 rows at once."

**Advanced cue:** Polars compiles the expressions in one `with_columns` call into
a single pass over the data — batch related derivations together.

**Key question:** "Why does the lease parse extract `^(\d+)` and not the whole
string?" Connect it to the two text formats in the data.

**Transition:** "Let us chain multiple operations together."

---

## Slide 22: Method Chaining

**Time:** ~3 min · Foundations

**Hook:** "Read this chain top to bottom — it is a sentence, not a program."

Method chaining is the Polars house style: each step transforms the frame and
hands the result to the next. Read the example as a sentence: "Take the data,
filter for Tampines, compute price per square metre, select these columns, sort,
show the top ten." Learners who can read a chain as prose can write one.

The parentheses trick — wrap the whole chain in parentheses so each method gets
its own line — is essential for readability and diff-friendliness. Show the same
logic written with intermediate variables first, then refactored into a chain,
so the equivalence is visible.

**Beginner cue:** If a chain confuses, break it into `df1 = …; df2 = …` steps,
verify each, then re-chain.

**Advanced cue:** The chain mirrors SQL structure — FROM, WHERE, computed SELECT,
ORDER BY, LIMIT — and Polars can optimise across the whole chain at once.

**Transition:** "Time to practise. Here is your exercise."

---

## Slide 23: Exercise 1.2: HDB Resale Filtering

**Time:** ~2 min (exercise work time ~10 min) · Foundations

**Hook:** "Five steps, one fluent chain by the end."

Walk the five steps on the slide — each maps to a method just learned: filter,
derive, select, sort, limit. The stretch goal is to express all five as one
chained expression; encourage it, but let beginners land the stepped version
first and verify each intermediate output.

One honest expectation to set: some filters in this exercise return empty
results on the synthetic data (for example, a 4-room flat in Ang Mo Kio under
S$500k does not exist in this dataset). An empty result is an answer, not a
crash — check the shape before assuming the code is wrong.

**Beginner cue:** Do each step separately, print the shape after each, then chain
once all five work.

**Advanced cue:** Early finishers add a date-range condition and a second sort
key — and should explain why the chain order changes the row count.

**Transition:** "Next lesson: we learn to write our own functions and compute
group-level statistics."

---

## Slide 24: Lesson 1.2 Recap

**Time:** ~1 min · Foundations

**Hook:** "Four verbs: filter, select, sort, with_columns."

These four methods cover roughly 80% of everyday data manipulation. Everything
else in the course composes them. Quick oral quiz: which verb picks rows? Which
adds a column? Which orders?

**Beginner cue:** If they can name what each of the four verbs does, they are
ready for Lesson 1.3.

**Advanced cue:** Natural break point before functions — take questions here.

**Transition:** "In Lesson 1.3, we learn to write reusable functions and
aggregate data by groups."

---

## Slide 25: Lesson 1.3: Functions and Aggregation

**Time:** ~2 min · Foundations

**Hook:** "Stop copy-pasting. Today you learn to write code that writes your
analysis for you."

This lesson bridges Python fundamentals — functions, lists, dictionaries, loops —
with the data-analysis pattern that matters most: `group_by` plus aggregation.
Functions make analysis reusable; instead of copy-pasting a filter-aggregate
block per town, you call a function with a town name.

**Beginner cue:** "A function is a recipe. You write it once, cook from it many
times."

**Advanced cue:** Higher-order functions and lambdas arrive in later modules; for
now, named `def` blocks with docstrings are the professional baseline.

**Transition:** "Let us define our first function."

---

## Slide 26: Defining Functions

**Time:** ~3 min · Foundations

**Hook:** "A vending machine: coins in, drink out. Parameters in, return value
out."

Walk the anatomy of `def` slowly — name, parameters, body, `return`. The
triple-quoted docstring documents what the function does; insist on it from day
one, because six weeks from now the docstring is the only documentation anyone
reads. Default parameters let callers omit arguments with sensible defaults —
show one call with the default and one overriding it.

**Beginner cue:** Trace one call end to end on the whiteboard: argument values
flow in, the body runs, the return value flows out.

**Advanced cue:** Mention type hints (`def price_per_sqm(price: float, area:
float) -> float`) — the course adds them as functions grow; they are
documentation the editor can check.

**Key question:** "What is the difference between printing inside a function and
returning a value?" The distinction matters the moment functions compose.

**Transition:** "Functions work on single values. Let us also work with
collections."

---

## Slide 27: Lists, Dictionaries, and Loops

**Time:** ~3 min · Foundations

**Hook:** "Lists are numbered checklists; dictionaries are phone books."

Lists are ordered and accessed by index — zero-based, which everyone off-by-ones
once. Dictionaries are accessed by key. `for` loops iterate over any collection,
and the pattern `for item in collection` reads like English by design.

Then the warning, in bold: Polars expressions are faster than loops for data
operations. Loops are for orchestration — calling a function per town — not for
arithmetic over a column. If a learner writes `for row in df.iter_rows()` this
week, gently redirect them to expressions.

**Beginner cue:** Live-code one loop that prints three town names, then show the
same result with a Polars expression — the contrast sticks.

**Advanced cue:** Preview list and dict comprehensions as the idiomatic shorthand
they will meet in the readings.

**Transition:** "Now let us combine functions with Polars to do group-level
aggregation."

---

## Slide 28: Group-By and Aggregation

**Time:** ~3 min · Foundations

**Hook:** "This one pattern — group, aggregate — answers most business questions
ever asked of data."

`group_by("town").agg(...)` answers "what is the average price per town?" — the
archetypal analytical question. Show that multiple aggregations live in one
`agg()` call — mean, median, row count, standard deviation — each with a clear
`.alias()`. The output is a new DataFrame with one row per group, which means it
chains like anything else.

**Beginner cue:** "Sort students into classrooms, then compute the average grade
per classroom. `group_by` sorts; `agg` computes."

**Advanced cue:** Polars group-by is multi-threaded — groups process in parallel,
which is why 50,000 rows aggregate in milliseconds.

**Key question:** "After grouping by town, how many rows does the result have —
and why 27?" Connect the output shape to the data's 27 towns.

**Transition:** "Let us see the available aggregation functions."

---

## Slide 29: Aggregation Functions

**Time:** ~3 min · Foundations

**Hook:** "Each of these collapses a pile of rows into one number. Choose the
number that answers your question."

Walk the table as column expressions — `pl.col("resale_price").mean()`,
`.median()`, `.std()`, and so on. Two correctness notes to say aloud: bare
`pl.mean()` needs a column name, so teach the expression form from the start;
and use `pl.len()` for row counts — the old `pl.count()` is deprecated and prints
a warning.

Multiple group keys give finer breakdowns — town and flat type together. And
`n_unique()` is the cardinality probe: how many distinct towns (27), flat types
(6), or months.

**Beginner cue:** "Mean is the average, median is the middle value, `pl.len()` is
how many rows. Start with those three."

**Advanced cue:** `quantile()` unlocks percentile analysis — P10/P90 bands are how
professionals bracket uncertainty on skewed prices.

**Transition:** "Let us write a reusable helper function that combines
everything."

---

## Slide 30: Reusable Analysis Functions

**Time:** ~3 min · Foundations

**Hook:** "One function call, a full district report. This is what reusable
means."

The function on screen encapsulates the whole mini-pipeline: filter to a
district, group by flat type, aggregate, sort. The type hints —
`df: pl.DataFrame, district: str` — document the contract. Call it for two
different towns live and let the room see identical structure, different numbers.

**Beginner cue:** "Give it a dataset and a district name; it hands back a summary
table. You never touch the inside again."

**Advanced cue:** Discuss the extension path: optional parameters for custom
aggregations, or a list of towns returning one concatenated report — this is the
seed of the capstone's pipeline structure.

**Transition:** "Here is the exercise for this lesson."

---

## Slide 31: Looping Over Districts

**Time:** ~3 min · Foundations

**Hook:** "Loop for orchestration, expressions for arithmetic — together they
scale."

The slide shows the loop-plus-function pattern: iterate over towns, call the
summary function, collect results. Two precision points to make. First,
`unique()` returns towns in no particular order, so `.sort()` makes "the first
five" reproducible rather than luck. Second, read the code's output label
carefully: the summary groups by flat type and sorts by average price
descending, so row zero is the town's *most expensive flat type* — not the town
average. The label must say what the number is.

Repeat the performance note: `group_by` is vectorised across the whole column;
loops are sequential over their iterations. Loop over 27 towns freely; never
loop over 50,150 rows.

**Beginner cue:** Focus on the pattern — iterate, call, collect. The fluency
comes with practice.

**Advanced cue:** A list comprehension is the more Pythonic collector here; `pl`
's `concat` then stacks the per-town frames into one report.

**Transition:** "Time for the exercise."

---

## Slide 32: Exercise 1.3: District-Level Statistics

**Time:** ~2 min (exercise work time ~10 min) · Foundations

**Hook:** "Write the function once. Answer for every district."

This exercise tests function writing and group-by aggregation together. The key
assessment criterion is reusability: functions must accept parameters — a town
name, a dataframe — not hardcode "TAMPINES" in the body. Circulate and check for
hardcoded strings; that is the tell of copy-paste thinking.

**Beginner cue:** Get the `group_by().agg()` working on the full data first, then
wrap it in `def`.

**Advanced cue:** Early finishers add 25th/75th percentile columns with
`quantile()` and interpret the spread per town.

**Transition:** "In Lesson 1.4, we learn to combine multiple datasets using
joins."

---

## Slide 33: Lesson 1.3 Recap

**Time:** ~1 min · Foundations

**Hook:** "Two tools earned today: functions and group-by."

Functions make code reusable; `group_by().agg()` compresses data into summaries.
Together they turn one-off analysis into a reporting tool.

**Beginner cue:** If they can write one working `def` and one working
`group_by().agg()`, the lesson has landed.

**Advanced cue:** Natural break point — take questions before joins.

**Transition:** "Lesson 1.4 introduces conditionals, imports, and joins for
combining multiple data sources."

---

## Slide 34: Lesson 1.4: Joins and Multi-Table Data

**Time:** ~2 min · Foundations

**Hook:** "No real dataset lives alone. Today your data makes friends."

Real-world analysis almost always spans multiple tables — transactions here,
stations there, schools somewhere else. This lesson adds two tools at once:
Python `if/else` for decisions in code, and joins for combining tables on a
shared key. The lesson's destination is an enriched HDB table with town-level
features.

**Beginner cue:** "Joining is merging two spreadsheets on a common column. You
have done it by eye; now it becomes code."

**Advanced cue:** Polars joins are parallel hash joins — a 50,000-row join takes
milliseconds, so the interesting problems are semantic (keys, cardinality), not
performance.

**Transition:** "First, let us learn Python's if/else for making decisions in
code."

---

## Slide 35: Conditional Statements

**Time:** ~3 min · Foundations

**Hook:** "Code that makes decisions — this is where programs stop being
calculators."

`if/elif/else` is Python's branching logic: the first condition that is True
wins, and order matters. Walk the price-tier example as a decision tree — read
each branch aloud as "if the price is over a million, luxury; otherwise if over
700 thousand, premium; otherwise…". Note the spelling: `elif`, not "else if" —
a Python-specific quirk that trips up learners coming from other languages.

**Beginner cue:** Indentation is syntax in Python — the block under each branch
must be indented consistently, or the program means something else.

**Advanced cue:** Row-level `if/else` inside a loop is the slow path; two slides
from now, `pl.when().then().otherwise()` does it vectorised.

**Key question:** "If the first two conditions are both True, which branch runs?"
Establish "first True wins" before the exercise.

**Transition:** "Now let us learn to import external packages."

---

## Slide 36: Imports and Packages

**Time:** ~3 min · Foundations

**Hook:** "Importing is plugging in an appliance — the capability arrives
instantly."

`import` makes external code available; `as` creates the alias everyone uses
(`polars as pl`). `from X import Y` pulls specific names and avoids the `X.`
prefix — show both forms in the same file and when each reads better. This is
also the moment to demystify the course's own `from shared import MLFPDataLoader`
— it is just an import of course-provided helper code.

**Beginner cue:** "If a name is not defined, you forgot an import. Read the
error's last line first — it names the missing thing."

**Advanced cue:** `__init__.py` and package structure come in later modules; for
now, the takeaway is that "packages are folders with code you can import".

**Transition:** "Now the main event: joining multiple DataFrames."

---

## Slide 37: Vectorised Conditionals in Polars

**Time:** ~3 min · Foundations

**Hook:** "The if/else you just learned, applied to 50,000 rows at once."

`pl.when().then().otherwise()` is the vectorised equivalent of `if/elif/else` —
same logic, evaluated on the entire column in one pass. Put the two side by side:
the Python loop version for readability, the Polars version for speed. Chained
`when` calls handle multiple tiers exactly like `elif`.

**Beginner cue:** "Same decision tree — it just runs on all rows simultaneously
instead of one at a time."

**Advanced cue:** `when/then` compiles into the same query plan as the
surrounding expressions — there is no hidden loop.

**Transition:** "Now let us learn about joins."

---

## Slide 38: Join Concepts

**Time:** ~3 min · Foundations

**Hook:** "Every join asks one question: which rows from the left deserve extra
columns from the right?"

Joins combine two tables on a shared key — here, `town`. Draw the Venn diagram:
inner is the overlap, left is the whole left circle, outer is both circles. Then
the rule of the course: **left join is the safe default for enrichment** — it
keeps every primary record — while an inner join silently drops unmatched rows,
which is a gotcha, not a feature.

Before any join, check two things about the key: does the *format* match (HDB
towns are UPPERCASE; the MRT and schools tables are Title Case), and is the key
*unique* on the right side (the MRT table has several stations per town — it is
one row per station, not per town). These two checks prevent the two classic
join disasters, which the next slide demonstrates live.

**Beginner cue:** The Venn diagram is the whole mental model — inner, left,
outer. Keep it visible.

**Advanced cue:** Polars also supports anti-joins ("rows with no match") and
cross joins — worth knowing exist.

**Transition:** "Let us see joins in action with Polars."

---

## Slide 39: Polars Joins in Practice

**Time:** ~3 min · Foundations

**Hook:** "Two traps. Both silent. Both on this slide."

Walk the two traps exactly as the code comment frames them. **Trap 1, case:** HDB
towns are UPPERCASE, MRT and school towns are Title Case — join raw and every one
of the 50,150 rows gets nulls, yet the row count still looks "correct", so the
failure hides. **Trap 2, fan-out:** the MRT table has 150 stations over 32 towns;
fix the case but skip aggregation and the join explodes to 186,997 rows — every
sale duplicated once per station in its town. The fix is the pattern in the code:
normalise the key with `str.to_uppercase()`, aggregate the right side to one row
per town, then `how="left"`, always explicit.

**Beginner cue:** After every join, print `.shape` and compare with the left
table. Row count changed unexpectedly? Stop and find out why.

**Advanced cue:** `left_on`/`right_on` handle keys with different names; also
mention validating join cardinality before production use.

**Key question:** "Why did the naive join return exactly 50,150 rows *and* zero
matches?" Make them articulate why row count alone proves nothing.

**Transition:** "What about rows that do not match?"

---

## Slide 40: Handling Missing Joins

**Time:** ~3 min · Foundations

**Hook:** "After every left join, your first move is `null_count()`."

Nulls after a left join are the unmatched rows made visible. On the course data,
6 HDB towns — 11,032 sales — have no station in the MRT table: BOON LAY, CENTRAL
AREA, HOUGANG, KALLANG/WHAMPOA, PUNGGOL, SENGKANG. Read the names with the room:
some are genuinely absent from the station table, others are *naming mismatches*
(KALLANG/WHAMPOA on one side, KALLANG on the other). `fill_null` papers over
absence but cannot fix a name mismatch — that takes an explicit mapping.

The `replace()` method with a dictionary is the quick way to remap values, as
the region-map example shows. And repeat the slide's caution: equal row counts
after a left join mean "right key unique" *or* "nothing matched" — only the null
count tells which.

**Beginner cue:** "Null means no data. Left join keeps the row and leaves a hole
where the match would be."

**Advanced cue:** Discuss one-to-many duplication and how `validate="m:1"` would
have caught the fan-out at join time.

**Transition:** "Here is your exercise for this lesson."

---

## Slide 41: Exercise 1.4: Multi-Table HDB Analysis

**Time:** ~2 min (exercise work time ~15 min) · Foundations

**Hook:** "Three tables in, one enriched dataset out — with no silent losses."

Set the expected solution shape: upper-case the town keys, aggregate MRT and
schools to one row per town, left-join onto HDB so all 50,150 records survive,
then check nulls. The features are honest town-level ones: **station count** as
the MRT-access proxy, typical **station spacing**, and the haversine distance
from each town's station centroid to the CBD — the feature that actually tracks
price (r ≈ −0.55).

Be direct about what the data cannot support: the MRT table's
`distance_to_mrt_km` column measures the gap between neighbouring *stations*, and
the HDB data has no flat coordinates — so a per-flat "distance to MRT" cannot be
computed from these tables. The old "walkability" framing was mislabelled data;
the exercise builds the honest version instead.

**Beginner cue:** Start with an inner join, look at the row count, then switch to
left and compare — the difference teaches more than the join itself.

**Advanced cue:** Challenge them to detect duplicate join keys programmatically
before joining.

**Transition:** "Lesson 1.5 takes us into time-series territory with window
functions."

---

## Slide 42: Lesson 1.4 Recap

**Time:** ~1 min · Foundations

**Hook:** "Joins are a daily operation — choose the type deliberately every
time."

Recap the flow: conditionals make decisions, imports bring in power, joins
combine tables, and null checks verify them. The one-sentence rule: left join is
the safe default for enrichment.

**Beginner cue:** "If you remember nothing else: upper-case the keys, one row per
town on the right, `how="left"`."

**Advanced cue:** Natural break point before time-series concepts — if running a
two-session delivery, this is the split.

**Transition:** "Lesson 1.5 introduces window functions and lazy frames."

---

## Slide 43: Lesson 1.5: Window Functions and Trends

**Time:** ~2 min · Foundations

**Hook:** "Aggregation collapsed your rows. What if you want the group statistic
*and* your rows?"

Window functions compute values across rows without collapsing them — the missing
tool between raw rows and group-by summaries. Frame the lesson's three stops:
`over()` for per-group enrichment, rolling windows for smoothing, and `shift()`
for period-over-period change. Lazy frames close the lesson as the Advanced
performance topic.

**Beginner cue:** "A window function is like writing each classroom's average
next to every student — everyone keeps their row."

**Advanced cue:** The lazy-evaluation slides at the end are where Polars' query
optimiser earns its keep — do not skip them if performance matters to you.

**Transition:** "Let us start with over(), the heart of window operations."

---

## Slide 44: Window Functions with over()

**Time:** ~3 min · Foundations

**Hook:** "Same data, two verbs: group_by shrinks the table, over() leaves it
exactly as tall."

The contrast is the whole slide: `group_by("town").agg(mean)` returns 27 rows —
one per town. `mean().over("town")` returns all 50,150 rows with the town mean
broadcast onto each. This enables row-versus-group comparisons — how far is this
flat's price from its town's average? — which is the seed of feature engineering.

**Beginner cue:** "`group_by` gives the class average as one number. `over()`
writes it next to every student."

**Advanced cue:** `over()` accepts multiple partition columns and even expressions
— per-town-per-year stats are one call away.

**Key question:** "After `.over("town")`, how many rows does the result have?"
50,150 — the answer anchors the broadcast intuition.

**Transition:** "Now let us apply this to time-series data with rolling
averages."

---

## Slide 45: Rolling Averages

**Time:** ~3 min · Theory

**Hook:** "Raw monthly prices jitter. A rolling average is how you see the
signal."

Rolling averages smooth noisy series — essential for spotting trends in property
prices. Two load-bearing details. First, `.over("town")` makes the window restart
per town; without it, one town's history bleeds into the next. Second, and
subtler: **windows count rows, not months**. The course town-by-month grid has
3,236 of 3,240 cells — BUKIT TIMAH has no sales in 2015-10, 2017-02 and 2022-07,
CENTRAL AREA none in 2015-08. Without completing the calendar first, a 3-row
window silently spans 4 months in those towns. The slide's grid-join code is the
fix: build every town × every month, left-join the data, then roll.

**Beginner cue:** Draw a 3-month sliding window on the whiteboard — current row
plus the two before it — and show why the first two rows per town are null.

**Advanced cue:** Exponentially weighted moving averages weight recent months
more heavily; the row-alignment problem applies to them identically.

**Key question:** "If a town is missing a month, what does a 3-row window
actually average?" Let them discover the 4-month span.

**Transition:** "Let us compute year-on-year changes."

---

## Slide 46: Year-on-Year Changes with shift()

**Time:** ~3 min · Theory

**Hook:** "Did prices rise or fall versus the same month last year? One shift
answers it."

`shift(12)` moves the column down by 12 rows — and that equals "same month last
year" **only because the calendar was completed on the previous slide**. Without
the grid, 40 rows compare against the wrong month — CENTRAL AREA 2016-03 would be
compared with 2015-02, and the "year-on-year" number would be fiction. With the
grid, a missing month yields a null YoY instead of a wrong one. Nulls are honest;
wrong numbers are not.

The formula is on screen: `(current − previous) / previous × 100`, rounded for
display. Positive means prices rose versus last year; negative means they fell.

**Beginner cue:** "YoY asks one question: compared to the same month last year,
up or down?"

**Advanced cue:** CAGR is the multi-year summary alternative — and equally
dependent on correct alignment.

**Transition:** "Let us test what patterns actually live in this series."

---

## Slide 47: Identifying Trends and Seasonality

**Time:** ~3 min · Theory

**Hook:** "Trend, seasonality, cyclicality, noise — every time series is these
four ingredients in different proportions."

Define the four components crisply: trend is long-term direction; seasonality is
a pattern repeating within each year; cyclicality is multi-year waves (in
Singapore property, cooling measures); noise is everything left over. Then the
professional habit: **seasonality is a hypothesis you test, not a pattern you
assume**. The slide's group-by on calendar month is the test.

Report the course-data result honestly: the 12 calendar-month averages sit within
about ±2% of each other (roughly S$875k in October to S$908k in September), and
monthly transaction counts are similar (about 4,000-4,300). No seasonal pattern.
The HDB file is synthetic, so it carries no real market seasonality — and with
real data, this exact code is how you would find it. A claimed "Chinese New Year
dip" must show up in the output before anyone says it aloud.

**Beginner cue:** "Trend: going up or down overall. Seasonality: same pattern
every year. That distinction is enough for today."

**Advanced cue:** Additive versus multiplicative decomposition arrives in Module
2 — the calendar-month group-by here is the poor-man's version.

**Key question:** "What would a real seasonal pattern look like in this output —
and what does ours show instead?"

**Transition:** "Now let us look at lazy frames for performance."

---

## Slide 48: Lazy Frames: Make It Faster

**Time:** ~3 min · Advanced

**Hook:** "Everything so far computed immediately. What if Polars planned the
whole computation first?"

Lazy frames load nothing until `.collect()` — instead Polars builds a query plan
and optimises it: skip unread columns, push filters into the scan, eliminate
redundant work. The code uses `loader.load_raw()` to get the local file path
(downloading first if needed) because `scan_parquet` needs a path, not a frame.

Label this slide honestly: it is Advanced material. Every exercise in the module
works in eager mode, and beginners may skip this without penalty.

**Beginner cue:** "Skim and move on — come back when a dataset is slow."

**Advanced cue:** Run `.explain()` on the lazy frame and read the plan together —
projection pushdown is visible in the output.

**Transition:** "Let us inspect the query plan."

---

## Slide 49: Inspecting the Query Plan

**Time:** ~3 min · Advanced

**Hook:** "`explain()` is Polars' X-ray — SQL veterans know it as EXPLAIN."

Read the plan bottom-up like SQL: the scan, the pushed-down filter, the projected
columns. The headline optimisation is projection pushdown — columns never
selected are never read from disk, which is why lazy pipelines on wide parquet
files can be dramatically faster than eager ones.

**Beginner cue:** Purely optional — note it exists and come back later.

**Advanced cue:** Streaming mode extends this to datasets larger than memory;
same plan, chunked execution.

**Transition:** "Time for the exercise."

---

## Slide 50: Exercise 1.5: HDB Price Trends

**Time:** ~2 min (exercise work time ~15 min) · Foundations

**Hook:** "Build the monthly series, complete the calendar, then smooth and
compare — the full trend workflow."

This exercise composes `group_by`, the calendar grid, `rolling_mean`, `shift`,
and sorting. Name the two pitfalls explicitly. Pitfall 1: `rolling_mean` and
`shift` must carry `.over("town")`, or towns contaminate each other. Pitfall 2:
`shift(12)` counts rows — complete the town × month calendar first, or YoY
compares the wrong months. That alignment is exactly what the module assessment
means by "proper time alignment".

**Beginner cue:** Build steps 1-3 first (monthly series, grid, rolling average);
tackle YoY and ranking only after those print sensibly.

**Advanced cue:** Early finishers rewrite the whole pipeline lazily and compare
`.explain()` plans.

**Transition:** "Lesson 1.5 recap."

---

## Slide 51: Lesson 1.5 Recap

**Time:** ~1 min · Theory

**Hook:** "Two concepts earned: window functions and lazy frames."

`over()` enriches rows without collapsing them; rolling and shift smooth and
compare — provided the calendar is complete. Lazy frames defer and optimise.
If one function survives the week, make it `over()`.

**Beginner cue:** "`over()` is the one to remember — group statistics on every
row."

**Advanced cue:** Natural break point before visualisation.

**Transition:** "Lesson 1.6 makes numbers visible."

---

## Slide 52: Lesson 1.6: Data Visualisation

**Time:** ~2 min · Foundations

**Hook:** "A chart is not decoration — it is a computation your eyes perform."

Visualisation is how you find patterns statistics hide and how you communicate
findings to people who do not read tables. The lesson runs why-before-how: the
case for charting, the attributes of good charts, the Gestalt principles, a
selection guide, and then the tooling — Plotly Express and Kailash
ModelVisualizer.

**Beginner cue:** "We start with *why* before *how* — thirty minutes of why saves
you years of misleading charts."

**Advanced cue:** The Gestalt and perceptual-ranking material is real design
theory — it will change how you review other people's dashboards.

**Transition:** "Why bother with charts when you have describe()?"

---

## Slide 53: Why Visualise?

**Time:** ~3 min · Foundations

**Hook:** "Four datasets. Identical means, identical variances, identical
correlations. Completely different shapes."

Anscombe's Quartet is the argument: summary statistics alone cannot distinguish
a clean line, a curve, a vertical scatter, and one outrageous outlier — only a
plot can. If the room has not seen it, show the four panels and let the gasp
happen. The lesson: `describe()` is necessary and never sufficient.

**Beginner cue:** Show the Quartet visually, not just the numbers — the visual
*is* the argument.

**Advanced cue:** The Datasaurus Dozen pushes the same point harder — a
tyrannosaur hiding in summary statistics.

**Key question:** "If two columns have the same correlation, can the relationship
still be completely different?" The Quartet is the permanent answer.

**Transition:** "What makes a chart good?"

---

## Slide 54: Attributes of Good Charts

**Time:** ~3 min · Foundations

**Hook:** "Four words: simple, clean, subtle, truthful."

Walk the four attributes with a counterexample for each — a 3-D exploding pie for
"simple", gridline noise for "clean", gratuitous colour for "subtle", a truncated
axis for "truthful". The truthful attribute deserves the most time: axis choices
and bin choices are editorial decisions, and they can lie.

**Beginner cue:** "When in doubt, use a bar chart or a line chart. The exotic
types earn their place rarely."

**Advanced cue:** Tufte's data-ink ratio formalises "clean" — every pixel should
carry information or be removed.

**Transition:** "Gestalt principles explain how our brains group visual
elements."

---

## Slide 55: Gestalt Principles

**Time:** ~3 min · Foundations

**Hook:** "Your brain groups what it sees before you think. Chart design exploits
that — or fights it."

Gestalt principles come from psychology: proximity, similarity, connection,
enclosure, and friends explain why some charts read instantly and others
confuse. Tie each principle to a chart choice — legend spacing (proximity),
colour coding (similarity), line continuity (connection). Designers use these
deliberately; analysts should at least not violate them accidentally.

**Beginner cue:** "Remember two: proximity and similarity. Things close together
and things that look alike are read as groups."

**Advanced cue:** Pre-attentive attributes — colour, size, position — are
processed in milliseconds; that is what makes a good dashboard scannable.

**Transition:** "Now let us pick the right chart for the right data."

---

## Slide 56: Chart Selection Guide

**Time:** ~3 min · Foundations

**Hook:** "The question chooses the chart — never the other way round."

Walk the table as a decision procedure: distribution → histogram; change over
time → line; comparison across categories → bar; relationship between two
numbers → scatter; composition → stacked bar; matrix of correlations → heatmap.
Have the room chant the mapping for two or three question types; it is the
single most reusable artefact of the lesson.

**Beginner cue:** "Histogram, line, bar — those three cover 80% of your needs
this year."

**Advanced cue:** Small multiples and faceting answer "same chart, per group" —
Plotly makes them one argument.

**Transition:** "Let us build these charts with Plotly Express."

---

## Slide 57: Plotly Express: Interactive Charts

**Time:** ~3 min · Foundations

**Hook:** "One function per chart type, and every chart is interactive out of the
box."

Plotly Express charts hover, zoom and pan with zero configuration — show it live;
interactivity is the moment learners realise these are not static pictures. And
one correction to say out loud because older tutorials get it wrong: **pass the
Polars DataFrame straight in**. Plotly 6 reads Polars natively — there is no
`.to_pandas()` step anywhere in this course.

**Beginner cue:** "Copy the pattern, change the column names. That is a
legitimate workflow for your first ten charts."

**Advanced cue:** Plotly Graph Objects is the lower-level API for fine control —
Express is built on it.

**Transition:** "Let us see how Kailash ModelVisualizer simplifies this."

---

## Slide 58: Kailash ModelVisualizer

**Time:** ~3 min · Foundations

**Hook:** "The second Kailash engine of the day — charts with the defaults
already right."

ModelVisualizer wraps Plotly with sensible ML defaults. The EDA-relevant methods
to teach by name and signature: `histogram(data, column, bins=, title=)`,
`scatter(data, x, y, color=, title=)`, and `box_plot(data, column, group_by=,
title=)`. Most of the engine's other methods serve models — `confusion_matrix`
takes `y_true`/`y_pred` label vectors, **not** a correlation matrix — which is
why the correlation heatmap on the next slide is built with Plotly directly.

It is Polars-native, so no conversion step exists. Position it honestly: a
shortcut for the standard charts; fall back to Plotly Express when you need
something custom.

**Beginner cue:** "It is a shortcut, not a cage. Custom chart? Plotly Express."

**Advanced cue:** The engine is extensible with custom themes — and note its
ExperimentalWarning on construction is a known upstream quirk, not a bug in
student code.

**Transition:** "Let us build a heatmap and stacked bar chart."

---

## Slide 59: Heatmaps and Stacked Bars

**Time:** ~3 min · Foundations

**Hook:** "Two chart types the selection guide promised: a correlation heatmap
and a stacked bar."

For the heatmap, compute the correlation matrix in Polars (`corr()`), then plot
with `px.imshow` on the numeric array. The colour scale matters: with `RdBu_r`
and `zmin=-1, zmax=1`, **red is positive, blue is negative, white is zero** — and
without the pinned `zmin/zmax`, the colour midpoint floats off zero and the chart
misleads. Read the course data together: floor area versus price is r = 0.47 —
a moderate relationship — while lease commencement year versus price is about 0;
the synthetic data carries no age effect, and the chart proves it.

The stacked bar shows composition per category — flat-type mix by town — and sets
up the exercise's sixth chart.

**Beginner cue:** Focus on reading these charts, not memorising the code —
where is the red? what does the tallest stack mean?

**Advanced cue:** Pearson correlation misses non-linear relationships — a zero on
this heatmap does not prove independence.

**Key question:** "Area explains r = 0.47, so about 22% of price variance. What
explains the other 78%?" Let them hypothesise — it previews feature engineering.

**Transition:** "Time for the exercise."

---

## Slide 60: Exercise 1.6: Six Charts from HDB Data

**Time:** ~2 min (exercise work time ~15 min) · Foundations

**Hook:** "Six chart types, one dataset, one per question from the selection
guide."

The exercise walks every chart type from the guide — histogram, scatter, bar,
line, heatmap, stacked bar. The assessment criterion to repeat: **no misleading
axes**. Titles must state the takeaway, axes must be labelled, and colour scales
must be pinned where zero matters.

**Beginner cue:** Start with the histogram and the bar — simplest first, momentum
matters.

**Advanced cue:** Early finishers add annotations and a deliberate colour
palette — and defend their choices in Gestalt terms.

**Transition:** "Lesson 1.6 recap."

---

## Slide 61: Lesson 1.6 Recap

**Time:** ~1 min · Foundations

**Hook:** "Visualisation is art *and* science — the science is the selection
guide, the art is the honesty."

Bookmark the chart selection guide; it is the lesson's takeaway artefact.
Tomorrow they will judge charts by it — including, happily, the instructor's.

**Beginner cue:** "Question first, chart second — you now have the map."

**Advanced cue:** Natural break before the Kailash engines arrive in force.

**Transition:** "Lesson 1.7 introduces DataExplorer for automated profiling."

---

## Slide 62: Lesson 1.7: Automated Data Profiling

**Time:** ~2 min · Foundations

**Hook:** "Everything you did by hand this morning — types, spreads, nulls,
duplicates — one engine call now does for you."

DataExplorer automates the manual inspection ritual from Lessons 1.1-1.3.
Position learners as *users* of the engine, not builders of it — the skill is
reading a profile and turning alerts into a cleaning plan. That plan-driven
framing is what separates profiling from sightseeing.

**Beginner cue:** "Think of it as a data-quality doctor — it reads the whole
chart and hands you a list of symptoms."

**Advanced cue:** The 8 alert types cover most automated data-quality checks you
would otherwise hand-roll; the thresholds are all configurable.

**Transition:** "Let us profile a dataset with one call."

---

## Slide 63: DataExplorer: One-Call Profiling

**Time:** ~3 min · Foundations

**Hook:** "One call: 50,150 rows read, every column summarised, every problem
flagged."

Run it live on the HDB data. `run_profile(df)` is the course's small sync wrapper
around DataExplorer's async `profile()` — it works identically in scripts and in
Colab/Jupyter, so students can use the engine before async is taught. Read the
headline numbers together: `n_rows` 50,150, `n_columns` 11, `duplicate_count`
300, and the alerts list.

Two API details to say precisely. Alerts are **dicts** with keys `type`, `column`
(or `columns` for a correlated pair), `value` and `severity` ("info" or
"warning") — access them as `alert["type"]`, not `alert.type`. And connect the
output back to the opening story: the high-skewness alert on `resale_price`
(skew 11.4) is exactly the S$10 / S$9M planted prices.

**Beginner cue:** Run the code, read the alerts aloud — that is the whole skill
for today.

**Advanced cue:** Inspect `profile.columns["resale_price"]`: `outlier_count` is
646 by the IQR rule even though no "outlier" alert exists — outliers are a
per-column statistic, not an alert type.

**Key question:** "Which alert would have caught the flash-crash scenario before
the dashboard shipped?" The skewness alert.

**Transition:** "What if the default thresholds are wrong for your data?"

---

## Slide 64: AlertConfig: Custom Thresholds

**Time:** ~3 min · Foundations

**Hook:** "Defaults are opinions. AlertConfig lets you argue with them."

Every `AlertConfig` argument is optional; omitted ones keep the defaults shown in
the slide's comments. Walk the direction rule: **lower thresholds mean more
alerts, higher mean fewer** — and make the concrete HDB example: `remaining_lease`
is 2.9% null, silent at the 0.05 default, flagged at 0.02. Note that moving the
null threshold *up* (say 0.10) *relaxes* the alert — direction mistakes are
common.

One field needs special care: `constant_threshold` is a **count of unique
values** (a column with at most 1 distinct value is constant), not a fraction —
a 0.99 there means something entirely different from what most people expect.

**Beginner cue:** "Defaults are fine to start. Tune only when an alert you
expected did not fire — or one fired that you do not care about."

**Advanced cue:** Outliers are still not an alert — each `ColumnProfile` carries
`outlier_count`/`outlier_pct` under the IQR rule for you to judge in domain
terms.

**Key question:** "There are exactly eight alert types — can the room name four?"
(high_nulls, constant, high_skewness, high_zeros, high_cardinality,
high_correlation, duplicates, imbalanced.)

**Transition:** "DataExplorer can also compare two datasets."

---

## Slide 65: Comparing Datasets

**Time:** ~3 min · Foundations

**Hook:** "Cleaning claims are cheap. `compare()` is the receipt."

`compare()` takes two **DataFrames** — not two profiles — and returns a dict:
`shape_comparison`, per-column `column_deltas`, both full `profile_a`/`profile_b`,
and `missing_in_a`/`missing_in_b` for schema drift. The course wrapper is
`run_compare(df_a, df_b)`.

The before/after on HDB is the teaching moment: after deduplicating and filtering
prices to a sane band, the skewness alert disappears (11.4 → 0.39) — and a **new**
high-correlation alert appears: floor area versus price, r = 0.91. The bad prices
were hiding the real relationship all along. Cleaning did not just remove noise;
it revealed signal.

**Beginner cue:** "It is a before-and-after photo of your data."

**Advanced cue:** For distribution shift beyond summary stats, KS tests and
Population Stability Index are the follow-up tools.

**Key question:** "Why did cleaning *add* an alert?" Make them connect removed
outliers to the unmasked correlation.

**Transition:** "Let us add error handling."

---

## Slide 66: Error Handling with try/except

**Time:** ~3 min · Foundations

**Hook:** "Programs that crash on bad input are demos. Programs that handle it
are tools."

`try/except` prevents crashes on *expected* errors — and the discipline is to
catch specific exception types, never a bare `except`. Demonstrate with the
loader: `MLFPDataLoader.load()` raises `FileNotFoundError` for a name it cannot
find — change the filename, watch the first `except` branch run, then read the
error message together. "Fail loudly, handle specifically" is the habit to
install.

**Beginner cue:** "Try this — but if it fails in *this specific way*, do that
instead."

**Advanced cue:** Custom exceptions and context managers are the next rung; the
rule against bare `except` never lifts.

**Transition:** "Exercise time."

---

## Slide 67: Exercise 1.7: Profile and Compare

**Time:** ~2 min (exercise work time ~15 min) · Foundations

**Hook:** "Three messy datasets, one profiler, and a cleaning plan you must
defend."

The exercise profiles three deliberately messy synthetic datasets modelled on
Singapore statistics — illustrative values throughout: mixed granularity, three
date formats inside one column, gaps, COVID-era outliers, and near-zero JPY
rates. The assessment asks for the *plan*, not just the alerts: read every alert,
decide the fix, justify it.

Set the division of labour honestly: the original-versus-cleaned `compare()` is
the capstone's job — Exercise 1.8 runs `run_compare(raw, cleaned)` end to end.
Here, the deliverable is profile literacy.

**Beginner cue:** Run the default profiler first, read *every* alert, and only
then decide how to clean.

**Advanced cue:** Write the cleaning as an automated function — profiling-driven
cleaning, not hand-patched.

**Transition:** "Lesson 1.7 recap."

---

## Slide 68: Lesson 1.7 Recap

**Time:** ~1 min · Foundations

**Hook:** "DataExplorer profiles; try/except protects. Together: robust
profiling."

One-call profiling with tunable alerts, dataset comparison as the cleaning
receipt, and specific error handling — the lesson in one breath.

**Beginner cue:** "DataExplorer plus try/except equals profiling you can trust."

**Advanced cue:** Natural break before the capstone.

**Transition:** "The final lesson brings everything together."

---

## Slide 69: Lesson 1.8: Data Pipelines and End-to-End Project

**Time:** ~2 min · Foundations

**Hook:** "Everything from the last seven lessons, assembled into one machine."

This is the capstone lesson. The pipeline is the morning's skills in order —
load, profile, clean, prepare, visualise, report — plus two production topics:
pulling data from REST APIs, and structuring a project so it can be rerun. The
destination is a complete exploratory data analysis on 50,000 messy taxi trips.

**Beginner cue:** "You already know every step. Today is about order and
automation, not new concepts."

**Advanced cue:** The API and project-structure segments are where professional
practice enters — pay attention even if the Polars feels familiar.

**Transition:** "Let us start with missing values."

---

## Slide 70: Handling Null Values

**Time:** ~3 min · Foundations

**Hook:** "Nulls are the most common defect in real data — and the most commonly
mishandled."

Three strategies, each with a cost. Drop the rows — simplest, but you lose data
and can bias the sample. Fill with a statistic (mean, median, zero) — keeps rows,
invents values; median is the robust default for skewed money. Forward/backward
fill — propagate neighbouring values, right for ordered series, wrong for
unordered ones. The professional question is never "how do I remove nulls" but
"why are they null, and what does each strategy claim about that?"

**Beginner cue:** "Null means *no data*. Dropping is simplest; filling is
smarter; knowing which is the job."

**Advanced cue:** MCAR/MAR/MNAR — the missingness mechanism decides whether
imputation is safe, and the data rarely tells you which one you have.

**Transition:** "Now let us extract data from APIs."

---

## Slide 71: REST APIs: Extracting Data

**Time:** ~3 min · Foundations

**Hook:** "So far every byte came from a file. Real pipelines pull from the
network."

REST APIs are how external services hand you data. The running example is
OneMap, Singapore's public mapping API, whose search endpoint needs no key.
`GET` asks for data — query parameters go in `params=`; `POST` sends data in the
request body — `requests.post(url, json={...})`. Both usually answer in JSON,
which `response.json()` turns into Python dicts and lists — the collections from
Lesson 1.3.

Two non-negotiables: always set a `timeout` so a slow server cannot hang the
pipeline, and always wrap API calls in `try/except` — networks fail and servers
go down; Lesson 1.7's error handling is what makes extraction production-safe.
Note for delivery: the live calls need network access; the printed counts on the
slide are examples, not guarantees.

**Beginner cue:** "An API is like ordering food — you send a request, you get a
response. Sometimes the kitchen is closed; plan for it."

**Advanced cue:** Pagination, rate limiting and async extraction are the
follow-up topics for large pulls.

**Transition:** "Now let us automate cleaning with PreprocessingPipeline."

---

## Slide 72: Kailash PreprocessingPipeline

**Time:** ~3 min · Foundations

**Hook:** "The third engine: a dishwasher for data — same cycle, every time."

`PreprocessingPipeline()` takes no constructor arguments. `setup(data=…,
target=…)` learns the preparation rules — imputation medians, category
inventories, scaling statistics — and returns a `SetupResult` carrying
`train_data`/`test_data`; `transform()` applies the same learned rules to new
rows later. `result.summary` reports what it did.

Two facts to state precisely. First, `setup()` **needs a target column** — it
prepares data for a model; it does not repair bad rows. Cleaning (negative
fares, duplicate IDs) happens *before* the pipeline, with the Polars skills from
this morning. Second, `setup()` learns from **every row it is given and splits
afterwards** — so hold out your test rows *first* and pass only training rows to
`setup()`, or test-set statistics leak into training. The shared helpers
(`split_then_preprocess` family) exist for exactly this.

**Beginner cue:** "Dishwasher: same cycle every time, no hand-scrubbing. But you
still scrape the plates first — that is the cleaning step."

**Advanced cue:** `pipeline.get_config()` returns the full configuration, so the
same preparation can be reproduced exactly — that is the reproducibility story.

**Key question:** "Why must the pipeline never see the test rows?" Make them say
"leakage" in their own words.

**Transition:** "The ETL pattern."

---

## Slide 73: The ETL Pattern

**Time:** ~3 min · Foundations

**Hook:** "Extract, Transform, Load — the assembly line every data team on earth
runs."

ETL is the standard data-engineering pattern, and the capstone is one: extract
from files and APIs, transform with the cleaning and feature code from this
morning, load the result to Parquet. Two details from the code: `Path` comes
from `pathlib` (imported back in Lesson 1.4), and `mkdir(parents=True,
exist_ok=True)` creates the output folder if missing. On formats: CSV is for
humans, **Parquet is for computers** — typed, compressed, and fast to scan, which
is why every course dataset ships as Parquet.

**Beginner cue:** "Raw materials in one end, finished product out the other —
and the line runs the same way every day."

**Advanced cue:** Airflow and Dagster orchestrate ETL at scale; the Kailash
workflow engine is the course's own path there in later modules.

**Transition:** "Project structure."

---

## Slide 74: Project Structure

**Time:** ~3 min · Foundations

**Hook:** "A pipeline you cannot rerun is a one-off. Structure is what makes it
a pipeline."

Walk the layout: `data/raw/` is sacred — **never modify raw data**; `data/processed/`
holds outputs; scripts are small and single-purpose; `main.py` orchestrates. The
reproducibility test to give every learner: delete the output folder, run
`main.py`, get the same results. If that fails, there is hidden state — a
hand-edited file, a hardcoded path — and it will fail in production too.

**Beginner cue:** "Copy this template for every project this year. Future you
says thanks."

**Advanced cue:** Cookiecutter templates and virtual environments standardise
this across teams.

**Transition:** "Capstone exercise."

---

## Slide 75: Exercise 1.8: Full EDA Pipeline

**Time:** ~2 min (exercise work time ~30-45 min) · Foundations

**Hook:** "50,000 taxi trips, six defect types, one pipeline. This is the module
in miniature."

Introduce the dataset honestly: a **synthetic** trip log built for this course —
the zone names and coordinates are realistic Singapore places, but the trips are
not real records — and it is deliberately dirty. Enumerate the planted defects so
the room knows the shape of the hunt: swapped GPS coordinates in 250 rows, 1,000
non-positive fares, 500 passenger counts below 1, 500 trips dated in the future
(2025-2027), 15 spellings of 4 payment methods, and 500 rows sharing trip_ids.

One technical landmine to defuse before it fires: Polars `dt.weekday()` is ISO
numbering — **Monday = 1 through Sunday = 7** — so the weekend filter is
`weekday >= 6`, and Friday is 5. Off-by-one here silently mislabels every chart.
And scope the API expectation: the exercise loads from a file; pulling extra
context from an API, as on the REST slide, is an extension, not a requirement.

**Beginner cue:** "Follow the ETL steps in order — extract, profile, clean,
prepare, visualise, report. The checklist on the slide is your map."

**Advanced cue:** Extensions: API enrichment, lazy evaluation of the whole
pipeline, and parameterising it into a reusable script per the project-structure
slide.

**Transition:** "Reference implementation."

---

## Slide 76: Capstone Pipeline Code

**Time:** ~3 min · Foundations

**Hook:** "Six blocks. Every block is a skill you earned today."

Read the reference implementation block by block and name the lesson each came
from: extract (1.1), profile with `run_profile` — 12 alerts on the raw taxi data
(1.7), clean with one auditable step per problem found (1.2, 1.4), compare
original versus cleaned with `run_compare` (1.7), prepare for a model with
PreprocessingPipeline (1.8), visualise with ModelVisualizer (1.6), and write the
HTML report with `run_report` (1.7). The wrappers — `run_profile`, `run_compare`,
`run_report` — are the shared sync shims around DataExplorer's async `profile`,
`compare` and `to_html`; they work identically in scripts and Colab.

Be transparent about scope: this is the short version for the slide. The full
`solutions/ex_8.py` adds GPS repair, timestamp parsing and future-date filtering,
payment-label normalisation, and the temporal/spatial feature engineering — and
it holds out test rows before `setup()`, as the leakage rule requires.

**Beginner cue:** "Copy this structure and modify it — that is how every pipeline
in industry starts."

**Advanced cue:** Parameterise the blocks into a reusable pipeline class; the
project-structure slide is the template.

**Key question:** "Which block would break first in production, and what protects
it?" Aim them at try/except around extract, and assertions after clean.

**Transition:** "Lesson 1.8 recap."

---

## Slide 77: Lesson 1.8 Recap

**Time:** ~1 min · Foundations

**Hook:** "Load, profile, clean, prepare, visualise, report — say it with me."

The pipeline order is the lesson. Recap slides make good study material — point
learners at the full series of them when revising for the assessment.

**Beginner cue:** "If you can name the six stages in order, you passed Lesson
1.8."

**Advanced cue:** Ready for the module summary.

**Transition:** "Module 1 summary."

---

## Slide 78: Module 1 Summary

**Time:** ~2 min · Foundations

**Hook:** "This morning you had never written Python. Read this slide — you now
own every item on it."

Walk both columns slowly and have learners tick items off mentally: variables
and f-strings; DataFrames, `describe()`, schemas; filter, select, sort,
`with_columns`; functions, loops, `group_by().agg()`; conditionals, imports,
joins with key hygiene; window functions on a complete calendar; the chart
selection guide and honest axes; DataExplorer profiles, alerts and comparisons;
null strategies, REST extraction, PreprocessingPipeline, ETL, project structure.
The callout bridges forward: Module 2 is statistical mastery for machine
learning — feature engineering and experiment design build directly on these
pipeline skills.

Close the loop with the opening question: "Can you trust a number you didn't
explore yourself?" They now have the tools to answer it — and the habit of
exploring first.

**Beginner cue:** "This slide is your study checklist for the assessment. Every
item on it is something you did with your own hands today."

**Advanced cue:** "Module 2 adds the statistical layer — distributions,
inference, experiment design — on top of exactly these pipelines."

**Transition:** "Thank you. See you in Module 2."
