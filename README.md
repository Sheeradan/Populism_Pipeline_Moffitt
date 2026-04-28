Populism Pipeline — Moffitt (2016)
A fully replicable research pipeline for classifying populist style in political TikTok videos. If you have a spreadsheet or CSV file with a list of TikTok video URLs, you can run this entire pipeline end to end — from raw URLs to publication-ready statistical figures.

What it does
The pipeline takes a list of TikTok URLs as its only input and produces coded, analysed, and visualised data across three analytical layers: performative register (how politicians present themselves visually), populist style (how they speak), and audience engagement (how viewers respond).

It was built for a BA Thesis analysing AfD politicians on TikTok using Benjamin Moffitt's (2016) populist style framework, but the design is general — you can apply it to any set of political TikTok accounts, any language (with a transcription model swap), and any grouping variable you define (party, country, gender, account tier, etc.).

The five stages
Stage 1 — Video download.
The script reads your URL list and downloads each video as an .mp4 file using the TikTok API (via RapidAPI). Videos that are photo slideshows rather than actual video are automatically detected and excluded. Everything else is kept.

Stage 2 — Audio extraction.
Audio is pulled from each .mp4 and saved as a standardised .wav file (16 kHz, mono, 16-bit PCM) using FFmpeg via MoviePy. The standardisation ensures consistent input for the transcription model.

Stage 3 — Transcription.
Each audio file is transcribed using WhisperX with the large-v2 model. The pipeline is configured for German but can be changed to any language Whisper supports. Output is both a plain-text transcript and a JSON file with word-level timestamps. The timestamps are preserved for optional prosody analysis.

Stage 4 — Prosody analysis (optional).
Using Parselmouth (a Python interface to Praat), the pipeline extracts acoustic features such as speech rate, pitch range, and intensity variation. These can be added as predictor variables in the engagement model.

Stage 5 — LLM classification.
This is the methodological core. Each transcript is classified for four binary Moffitt (2016) populist style variables using a locally running large language model (Gemma 3 27b via Ollama — no API costs, no data leaves your machine). The classification runs twice with two differently structured prompts:

Pass A presents full theoretical definitions first, then asks for scores.
Pass B opens with known error patterns and calibration rules, then reverses the variable order to stress-test the hardest variable (Bad Manners) first.
Where both passes agree, the code is accepted. Where they disagree, the case is flagged automatically for manual review against the original video. This two-pass design gives you a built-in reliability check without needing a second human coder. The script checkpoints every 10 videos, so a crash or interruption never loses more than 10 videos of work.

What gets coded
Populist style (LLM-coded):

Appeal to the People — does the speaker construct a shared "people" and position themselves as one of them?
Anti-Elitism — does the speaker explicitly name and delegitimise an elite antagonist?
Bad Manners — does the speaker use mockery, sarcasm, colloquial register, hyperbolic labels, or performative outrage in a political context?
Crisis / Breakdown / Threat — does the speaker frame the current situation as an emergency, systemic collapse, or existential threat?
Performative register (human-coded, entered in your spreadsheet):

Dress code (formal vs. informal)
Setting (studio/office vs. casual/outdoor)
Production quality (professional vs. smartphone)
Video format (talking-head vs. other)
Engagement (calculated from your spreadsheet data):

A weighted engagement score Q = Likes + (Comments + Shares + Saves) × 2.5
An engagement rate ER = (Q / Views) × 100, which controls for reach
What the analysis produces
Once coding is complete, the pipeline runs three sets of analyses and generates all figures automatically:

RQ1 — Register analysis. Chi-square tests and Fisher's exact tests compare register coding across your defined groups (tiers, parties, etc.). Effect sizes are reported as Cramér's V. Output includes a summary table, faceted bar charts, a heatmap, a pairwise comparison matrix, and an effect size plot.

RQ2 — Populist style analysis. The same statistical approach applied to the four Moffitt variables. Additional figures include a diverging lollipop chart showing deviation from the sample mean, a co-occurrence bar chart, and a paradox figure that places RQ1 and RQ2 results side by side on a shared scale — making the contrast between style segmentation and content uniformity visually immediate.

RQ3 — Engagement analysis. Because engagement data is heavily skewed, the pipeline uses non-parametric tests throughout: Kruskal-Wallis H for overall group differences, Mann-Whitney U for pairwise comparisons (Bonferroni-corrected), and rank-biserial r as the effect size. Results are verified against any manually computed table values you provide. Figures include raincloud plots (half-violin + box + strip), an informality scatter plot with per-group trend lines, a heatmap of median engagement by register coding and group, and a two-panel composite showing informal usage rates alongside their engagement payoff.

Pipeline diagrams. Two flow chart figures are generated for the methodology section of a thesis or paper: one for the automated pipeline and one for the manual validation process.

What you need to get started
A spreadsheet (CSV or Excel) with at least one column of TikTok URLs and a unique video ID per row. If you want engagement analysis, you also need Views, Likes, Comments, Shares, and Saves columns — all available from TikTok Analytics or any scraper.
A RapidAPI key for the TikTok scraper (free tier covers small corpora).
Python 3.11+ with a virtual environment.
Ollama installed locally with gemma3:27b pulled (~17 GB download, runs on a modern GPU or slowly on CPU).
FFmpeg installed on your system.
A CUDA-capable GPU for WhisperX (strongly recommended; transcription on CPU is very slow).
Everything else — all statistical tests, all figures — runs on standard Python scientific libraries (pandas, scipy, statsmodels, matplotlib, seaborn) with no paid services.

How to adapt it
Different language: change the language parameter in the WhisperX call and translate the classification prompts.
Different grouping variable: replace the tier assignment function with your own logic (party affiliation, country, gender, time period — anything in your spreadsheet).
Different theoretical framework: the prompts in Script 05 are the only thing that's framework-specific. Rewrite them for any binary coding scheme and the rest of the pipeline works unchanged.
Larger corpus: the checkpoint system means you can run the LLM classifier overnight across hundreds of videos without supervision.
