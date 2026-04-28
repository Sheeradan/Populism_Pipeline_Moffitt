"""
04_llm_classify.py
---------------------
Two-pass Moffitt populist style classifier using Ollama (Gemma 3 27b).

Architecture (see thesis Section 3.5.3):
  Pass A  — definition-first prompt  (Appeal → Anti-Elitism → Bad Manners → Crisis)
  Pass B  — decision-rule-first prompt (Bad Manners → Crisis → Appeal → Anti-Elitism)
  Merge   — agree → accept value; disagree → flag for manual review

The two-pass design tests prompt sensitivity: where both passes converge,
the code is accepted; where they diverge, the case is flagged for manual
review against the original video recording.

Output (results/):
  pass_a_<model>_<date>.csv   — Pass A raw classifications with reasoning
  pass_b_<model>_<date>.csv   — Pass B raw classifications with reasoning
  merged_<model>_<date>.csv   — Merged output (disagreement column lists diverging vars)

Dependencies:
  pip install pandas requests tqdm
  Ollama must be running locally with gemma3:27b pulled.

Usage:
  python 04_llm_classify.py
  python 04_llm_classify.py --model gemma3:27b --skip-a  # resume from existing Pass A
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:27b"
TEMPERATURE = 0.1          # Low temperature for classification consistency
TIMEOUT = 180              # seconds; increased for reasoning output
TRANSCRIPTS_DIR = Path("transcripts")
METADATA_CSV = Path("Data.csv")
RESULTS_DIR = Path("results")
SAVE_EVERY = 10            # Checkpoint interval

VARS = [
    "Appeal_to_the_People",
    "Anti_Elitism",
    "Bad_Manners",
    "Crisis_Breakdown_Threat",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ===========================================================================
# PROMPT A — Definition-first
# Order: Appeal to the People → Anti-Elitism → Bad Manners → Crisis
#
# This prompt presents the four variable definitions in their theoretical
# order (Moffitt 2016), followed by German-language lexical examples for
# each score value and a list of known boundary cases.
# ===========================================================================
PROMPT_A_TEMPLATE = """\
You are classifying a German-language TikTok transcript for four binary \
populist style variables following Moffitt (2016).

## Variable Definitions

### 1. Appeal_to_the_People (Moffitt Dimension 1, people-centric side)
Following Farkas & Bene (2022): Closeness and Ordinariness sub-dimensions.

Score 1 if the transcript contains a performative construction of "the people" \
and the speaker's claimed proximity to them:
- Direct address constructing a shared in-group: "Leute", "Freunde", \
"liebe Bürger", inclusive "wir" (speaker + audience as shared group)
- Self-presentation as ordinary person / one of us: "ich als normaler Bürger", \
"wir alle kennen das", shared everyday experience
- Audience inclusion techniques: "schreibt mir", "was meint ihr", \
"sagt mir eure Meinung"
- National identity as inclusive bond: "wir Deutschen", "unser Land" — \
ONLY when constructing shared belonging, NOT when excluding out-groups

Score 0 if:
- Speaker discusses policy without constructing "the people" as in-group
- "Du" addresses interviewer or opponent rather than the audience as "the people"
- Speech is about third parties without people-construction
- Personal anecdote without connection to shared popular experience

IMPORTANT: "du" is standard TikTok platform convention. By itself it does NOT \
trigger Appeal_to_the_People. The variable requires performative construction \
of a collective "people."

Additional rule: When a speaker repeats "wir wollen" + policy goals, they are \
constructing an inclusive political subject ("we the people want X") and score 1, \
even if the phrasing sounds formal. The "wir" must construct a speaker-audience \
bond, not merely refer to the party: "wir als AfD" = party reference = 0; \
"wir in diesem Land" = people-construction = 1.

### 2. Anti_Elitism (Moffitt Dimension 1, anti-elite side)
Following Farkas & Bene (2022): Elites sub-dimension.

Score 1 if the transcript contains a performative construction of "the elite" as \
antagonist and delegitimization of the political establishment:
- EXPLICIT naming of elite antagonists: "die Ampel", "die Regierung", "Scholz", \
"Habeck", "Baerbock", "die Grünen", "die Altparteien", "die da oben", \
"Berliner Blase", "Brüssel"
- Delegitimization beyond policy critique: framing opponents as corrupt, \
incompetent, out of touch, or morally bankrupt
- Media as elite enemy: "Lügenpresse", "Systemmedien"
- Moral monopoly: only we truly represent the people

Score 0 if:
- Speaker criticizes a specific POLICY without naming a responsible elite antagonist
- Speaker presents own positive positions without attacking opponents
- Criticism is analytical and factual rather than delegitimizing \
(e.g., "die CDU hat dagegen gestimmt" as fact, not performative attack)

CRITICAL DISTINCTION: Appeal_to_the_People constructs the IN-GROUP. \
Anti_Elitism constructs the OUT-GROUP. Both can co-occur.

### 3. Bad_Manners (Moffitt Dimension 2)
Moffitt (2016, p. 44): "a coarsening of political rhetoric and disregard for \
'appropriate' ways of acting in the political realm." Ostiguy (2009): \
"flaunting of the low."

THRESHOLD CALIBRATION: The threshold for Bad_Manners is LOWER than you think. \
Moffitt's concept is broad: any "coarsening of political rhetoric" or "disregard \
for appropriate ways of acting in the political realm" (2016, p. 44) counts. \
You do NOT need swearing or explicit vulgarity. Mockery, sarcasm, hyperbolic \
labels, rhetorical provocations, and colloquial register in political contexts \
suffice. If an experienced Bundestag observer would consider the language \
inappropriate for a plenary speech, it is Bad Manners.

Score 1 if the transcript shows ANY of the following:
(a) Linguistic transgression: slang, swearing, crude insults, dehumanizing \
nicknames for opponents, mockery as rhetorical strategy
(b) Register transgression: informal "du" toward political opponents or \
institutional actors signals deliberate disrespect. NOTE: "du" toward \
TikTok audience is standard and does NOT count.
(c) Political incorrectness as virtue: "das wird man ja wohl noch sagen dürfen", \
deliberately provocative statements violating social taboos
(d) Performative mockery: sarcasm, ridicule, theatrical outrage, derision of \
opponents — not factual criticism but performative contempt
(e) Hyperbolic provocation: "Ich bin der letzte Deutsche" — deliberately \
sensationalist framing to shock
(f) Hyperbolic political labels as rhetorical weapons: "Migrantengeld" \
(instead of Bürgergeld), "Zwangsgebühr" (instead of Rundfunkbeitrag), \
"CDU-Show", "geifernder Kindergarten" — these deliberate renamings of \
institutions/policies with delegitimizing labels are textbook Moffitt: \
populists lower political discourse through "exaggerated claims" and \
"taunts" (2016, p. 60).
(g) Rhetorical questions as mockery: "Möchte die CDU dir auch noch den Urlaub \
verbieten?" or "Was machen Sie hier eigentlich?" — when a rhetorical question \
is designed to ridicule rather than genuinely ask, it is performative mockery.
(h) Performative outrage in colloquial register: When a speaker discusses a \
political event (crime, policy decision, scandal) with visible dramatization — \
exclamatory syntax, escalating repetition, alarm register — AND in colloquial \
rather than institutional German, this combination triggers BM=1. The question \
is not whether the event is serious, but whether the speaker performs outrage \
in a register that violates norms of measured political conduct.
(i) Sarcastic or ironic framing: "Ich wünschte mir, dass das Blödsinn wäre oder \
Fake News, aber es ist Realität" — the sarcastic setup before a political claim \
is a performative technique that violates institutional norms of straightforward \
argumentation.
(j) Colloquial directness in institutional critique: When a politician uses \
"du"-register, everyday metaphors, or street language to discuss institutional \
topics (budgets, court decisions, parliamentary votes), this register breach \
IS the transgression. Formal topic + informal register = Bad Manners. \
This is Ostiguy's "low" in action.

Score 0 if:
- Speaker makes radical or extreme CLAIMS but delivers them in formal, calm, \
institutional register (measured pace, parliamentary vocabulary, no mockery). \
Radical content in genuinely formal style = 0.
- Motivational or inspirational tone, even if informal, without transgression
- Analytical factual criticism without mockery or contempt

Style-content independence rule (calibrated): Style-content independence still \
applies — a speaker making extreme claims in genuinely formal, composed, \
institutional German (measured pace, Bundestag vocabulary, no mockery) scores 0. \
But most AfD TikTok content does NOT use genuinely formal register — if the \
speaker uses colloquial German, mockery, hyperbole, sarcasm, or performative \
outrage in political claims, score 1 regardless of whether the underlying \
content is radical or moderate.

### 4. Crisis_Breakdown_Threat (Moffitt Dimension 3)
Moffitt (2016, p. 49): performative articulation of crisis — framing the \
situation as immediate danger, systemic failure, or existential threat.

CALIBRATION: Crisis framing does NOT require the word "Krise." Moffitt (2016) \
describes it as "the performance of crisis, breakdown or threat" — i.e., the \
speaker CONSTRUCTS a sense of emergency, danger, or systemic collapse through \
framing choices. Accumulated problem lists, before/after decline narratives, \
conditional future threats, and instrumentalization of events as evidence of \
systemic failure ALL count as crisis framing.

Score 1 if the transcript frames the current situation as crisis, breakdown, \
or existential threat:
- Explicit crisis vocabulary: "Krise", "Zusammenbruch", "Katastrophe", \
"Notstand", "es ist fünf nach zwölf", "dieses Land geht kaputt"
- Systemic failure framing: "nichts funktioniert mehr", "der Staat versagt", \
"totales Versagen"
- Threat construction: migration as existential threat, loss of sovereignty, \
cultural dissolution
- Urgency/alarm: "sofort handeln", "letzte Chance", "jetzt oder nie"
- Implicit systemic failure through accumulation: When a speaker lists multiple \
problems (crime, migration, economic decline, cultural change) and links them \
into a pattern of systemic failure — even without the word "Krise" — this is \
crisis framing. "Wir haben die höchsten Energiepreise, Betriebe schließen, \
Kriminalität steigt, Grenzen sind offen" = crisis framing by accumulation.
- Comparative decline frame: "Früher war es normal, dass du zweimal im Jahr \
in den Urlaub fahren konntest. Und wie ist es jetzt?" — the before/after \
structure constructing a narrative of national decline IS crisis framing, \
even without explicit alarm vocabulary.
- Conditional threat framing: "In drei Jahren nimmt man dir dein Auto weg" or \
"Wenn wir nicht handeln, dann..." — projection of future harm as consequence \
of current policy IS threat construction per Moffitt.
- Instrumentalization of events: When a speaker takes a concrete incident \
(a crime, an attack, a policy decision) and frames it as EVIDENCE of a broader \
pattern of breakdown — "das ist immer wieder das gleiche Schema" — this \
instrumentalization is crisis framing, even if the speaker never uses explicit \
crisis vocabulary.
- Policy discussion with existential stakes: When policy criticism is framed \
not as "this could be better" but as "this threatens our existence/freedom/future" \
— the existential escalation triggers Crisis=1. "Deutschland wird als \
Wirtschaftsstandort bald Geschichte sein" is crisis framing, even if delivered \
in measured language.

Score 0 if:
- Speaker criticizes a specific policy without systemic-crisis framing
- Speaker discusses problems proportionally without urgency, alarm, or \
collapse framing
- Speaker makes a positive policy proposal without crisis backdrop
- Biographical or introductory content without crisis framing
- Strong words without existential dimension: "Zwangsgebühr" about broadcasting \
= complaint, not crisis

CRITICAL DISTINCTION: Anti_Elitism names WHO is responsible (the elite). \
Crisis frames WHAT is happening (everything is collapsing). Both can occur \
independently.

## Transcript

{transcript}

## Task

Classify each variable as 0 (ABSENT) or 1 (PRESENT).
Respond with a JSON object containing the four binary scores AND a brief \
reasoning string (1-2 sentences per variable) explaining your decision. \
No other text outside the JSON.

Format:
{{"Appeal_to_the_People": 0, "Anti_Elitism": 0, "Bad_Manners": 0, \
"Crisis_Breakdown_Threat": 0, "reasoning": "Appeal_to_the_People: [reason]. \
Anti_Elitism: [reason]. Bad_Manners: [reason]. Crisis_Breakdown_Threat: [reason]."}}
"""


# ===========================================================================
# PROMPT B — Decision-rule-first
# Order: Bad Manners → Crisis → Appeal → Anti-Elitism
#
# This prompt reverses the approach: it opens with explicit decision rules
# and common error patterns, places Bad Manners first in the variable
# sequence to allocate the model's strongest attention to the most difficult
# variable, and substitutes different boundary examples.
# ===========================================================================
PROMPT_B_TEMPLATE = """\
You are classifying a German-language TikTok transcript for four binary \
populist style variables following Moffitt (2016).

READ THESE DECISION RULES FIRST, THEN THE DEFINITIONS:

## Known Error Patterns from Prior Classification Runs

1. BAD_MANNERS is UNDERCODED: The most common error is assigning 0 when the \
transcript actually contains mockery, sarcasm, hyperbolic labels, or colloquial \
register that violates political norms. Provocative CONTENT is NOT the same as \
transgressive STYLE — but most AfD TikTok content does NOT use genuinely formal \
register. If the speaker uses colloquial German, mockery, sarcasm, hyperbolic \
labels, or performative outrage, score 1.

2. ANTI_ELITISM is undercoded: Requires EXPLICIT naming of an elite antagonist. \
Mere policy criticism, even if intense, does not count.

3. CRISIS is UNDERCODED: The model misses implicit crisis framing. You do NOT \
need the word "Krise." Accumulated problem lists, before/after decline narratives, \
conditional future threats, and instrumentalization of events as evidence of \
systemic failure ALL count. "Zwangsgebühr" about broadcasting fees = complaint, \
not crisis. "Mit Deutschland geht es bergab" IS crisis framing. Also: \
accumulating multiple problems into a systemic decline narrative = crisis framing.

4. APPEAL_TO_THE_PEOPLE: "du" on TikTok is standard convention. Requires \
performative construction of a collective "people," not just second-person \
address. "wir wollen" + policy goals repeated = people-centric construction = 1 \
(if "wir" means speaker + audience, not just the party).

---

## CALIBRATION CHECK for Bad_Manners

The most common error is UNDERCODING — assigning Bad_Manners=0 when the \
transcript contains mockery, sarcasm, hyperbolic labels, or colloquial register \
that violates political norms. Before you assign Bad_Manners=0, ask yourself: \
Would this language be acceptable in a formal Bundestag speech by a mainstream \
politician? If the answer is no — if a CDU or SPD politician would be criticized \
for speaking this way — score 1.

---

## Variable Definitions

### 1. Bad_Manners (Moffitt Dimension 2) — MOST DIFFICULT VARIABLE

Moffitt (2016, p. 44): "a coarsening of political rhetoric and disregard for \
'appropriate' ways of acting." Ostiguy (2009): "flaunting of the low."

THRESHOLD CALIBRATION: The threshold is LOWER than you think. You do not need \
swearing or explicit vulgarity. Mockery, sarcasm, hyperbolic labels, rhetorical \
provocations, and colloquial register in political contexts suffice. If a \
Bundestag observer would consider the language inappropriate for a plenary \
speech, it is Bad Manners.

Score 1 if the transcript shows ANY of the following:
(a) Linguistic transgression: slang, swearing, crude insults, dehumanizing \
nicknames, mockery as strategy
(b) Register transgression: informal "du" toward political opponents or \
institutional actors (NOT toward TikTok audience)
(c) Political incorrectness as virtue: "das wird man ja wohl noch sagen dürfen", \
taboo violations as performance
(d) Performative mockery: sarcasm, ridicule, theatrical outrage, derision
(e) Hyperbolic provocation: "Ich bin der letzte Deutsche" — deliberately \
sensationalist to shock
(f) Hyperbolic political labels as weapons: "Migrantengeld", "Zwangsgebühr", \
"CDU-Show" — delegitimizing renaming of institutions/policies
(g) Rhetorical questions as mockery: designed to ridicule rather than \
genuinely ask
(h) Performative outrage + colloquial register: dramatization (exclamatory, \
escalating, alarm register) IN colloquial German
(i) Sarcastic/ironic framing before political claims
(j) Colloquial directness on institutional topics: formal topic + informal \
register = transgression

Score 0 if:
- Extreme CLAIMS in genuinely formal, composed register without mockery \
(radical content ≠ bad manners — but this is rare in AfD TikTok)
- Motivational tone without transgression
- Factual criticism without mockery or contempt

Boundary cases:
- "Die Ampel zerstört Deutschland" in calm, serious tone = 0 \
(extreme content, normal register)
- "Diese inkompetenten Idioten in Berlin" with contempt = 1 \
(linguistic transgression + performative contempt)
- Sarcastic laughter while quoting an opponent = 1 (performative mockery)
- "Migrantengeld" instead of Bürgergeld = 1 (hyperbolic delegitimizing label)
- "Ich wünschte mir, das wäre Fake News, aber es ist Realität" = 1 \
(sarcastic setup)

### 2. Crisis_Breakdown_Threat (Moffitt Dimension 3)

Performative articulation of crisis, systemic failure, or existential threat.

CALIBRATION: "Krise" is NOT required. Moffitt describes it as "the performance \
of crisis, breakdown or threat" — the speaker constructs emergency/danger/systemic \
collapse through framing. Accumulation, before/after decline, conditional threats, \
and event instrumentalization ALL count.

Score 1 if:
- Explicit crisis vocabulary: "Krise", "Zusammenbruch", "Katastrophe", \
"es ist fünf nach zwölf", "dieses Land geht kaputt"
- Systemic failure: "nichts funktioniert mehr", "der Staat versagt"
- Migration as existential threat / sovereignty loss / cultural dissolution
- Urgency: "letzte Chance", "jetzt oder nie", "sofort handeln"
- Implicit crisis accumulation: multiple problems linked into systemic \
decline narrative, even without "Krise" word
- Comparative decline: "Früher konntest du zweimal im Jahr in den Urlaub. \
Und jetzt?" — before/after decline narrative
- Conditional threat: "In drei Jahren nimmt man dir dein Auto weg" / \
"Wenn wir nicht handeln..."
- Instrumentalization: concrete incident framed as evidence of systemic \
failure — "das ist immer wieder das gleiche Schema"
- Existential policy critique: not "this could be better" but "this threatens \
our existence/freedom/future"

Score 0 if:
- Policy critique without crisis framing
- Proportional problem discussion without alarm/collapse framing
- Positive policy proposal without crisis backdrop
- Strong words without existential dimension: "Zwangsgebühr" about \
broadcasting = complaint, not crisis

Boundary cases:
- "Die Rundfunkgebühren sind ungerecht und müssen weg" = 0 (complaint, not crisis)
- "Mit Deutschland geht es bergab, unser Land braucht eine echte Wende" = 1 \
(crisis construction)
- Multiple problems accumulated without explicit crisis word but with decline \
narrative = 1
- "Deutschland wird als Wirtschaftsstandort bald Geschichte sein" = 1 \
(existential escalation, even in measured language)

### 3. Appeal_to_the_People (Moffitt Dimension 1, people-centric side)

Performative construction of "the people" and claimed proximity of the speaker.

Score 1 if:
- In-group construction: "Leute", "Freunde", "liebe Bürger", inclusive "wir" \
(speaker + audience)
- Self-presentation as one of us: "ich als normaler Bürger", shared everyday \
experience
- Audience inclusion: "schreibt mir", "was meint ihr"
- National identity as inclusive bond: "wir Deutschen" (ONLY when constructing \
belonging, not when excluding out-groups)
- "wir wollen" + policy goals repeated: constructs inclusive political subject, \
even if formal-sounding. Check the "wir": "wir als AfD" = party reference = 0; \
"wir in diesem Land" / "wir alle" = people-construction = 1.

Score 0 if:
- Policy discussion without people-construction
- "du" = platform convention only, no collective people constructed
- Speech about third parties without people-construction
- Personal anecdote without people-reference

### 4. Anti_Elitism (Moffitt Dimension 1, anti-elite side)

Performative construction of "the elite" as antagonist and delegitimization \
of the establishment.

Score 1 if:
- EXPLICIT naming: "die Ampel", "die Regierung", "Scholz", "Habeck", \
"Baerbock", "die Grünen", "die Altparteien", "die da oben", "Berliner Blase", \
"Brüssel"
- Delegitimization: framing opponents as corrupt, incompetent, out of touch, \
morally bankrupt
- Media as enemy: "Lügenpresse", "Systemmedien"

Score 0 if:
- Policy critique without named elite antagonist
- Own positive positions without opponent attack
- Analytical factual statement ("die CDU hat dagegen gestimmt" as fact)

---

## Transcript

{transcript}

## Task

Classify each variable as 0 (ABSENT) or 1 (PRESENT).
Respond with a JSON object containing the four binary scores AND a brief \
reasoning string (1-2 sentences per variable) explaining your decision. \
No other text outside the JSON.

Format:
{{"Appeal_to_the_People": 0, "Anti_Elitism": 0, "Bad_Manners": 0, \
"Crisis_Breakdown_Threat": 0, "reasoning": "Bad_Manners: [reason]. \
Crisis_Breakdown_Threat: [reason]. Appeal_to_the_People: [reason]. \
Anti_Elitism: [reason]."}}
"""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def call_ollama(prompt: str, model: str, logger: logging.Logger) -> str | None:
    """Send a prompt to Ollama and return the raw response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.error("Ollama request failed: %s", exc)
        return None


def parse_output(raw: str | None, vid_id, logger: logging.Logger) -> dict | None:
    """
    Parse the LLM response into a dict with four binary scores and reasoning.
    Attempts direct JSON parse first, then regex fallback for robustness.
    """
    if not raw:
        logger.error("Empty response for ID %s", vid_id)
        return None

    # Try direct JSON parse
    try:
        data = json.loads(raw)
        result = {v: int(data[v]) for v in VARS}
        result["reasoning"] = data.get("reasoning", "")
        return result
    except Exception:
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            result = {v: int(data[v]) for v in VARS}
            result["reasoning"] = data.get("reasoning", "")
            return result
        except Exception:
            pass

    # Try finding any JSON object in the response
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            result = {v: int(data[v]) for v in VARS}
            result["reasoning"] = data.get("reasoning", "")
            return result
        except Exception:
            pass

    # Last-resort regex fallback for individual variables
    result = {}
    for var in VARS:
        m = re.search(rf'"{var}"\s*:\s*([01])', raw)
        if m:
            result[var] = int(m.group(1))
    if len(result) == len(VARS):
        # Extract reasoning if present
        r = re.search(r'"reasoning"\s*:\s*"(.*?)"', raw, re.DOTALL)
        result["reasoning"] = r.group(1) if r else ""
        return result

    logger.error("Parse failed for ID %s — raw: %.300s", vid_id, raw)
    return None


# ---------------------------------------------------------------------------
# Single-pass runner
# ---------------------------------------------------------------------------
def run_pass(
    pass_name: str,
    prompt_template: str,
    ids: list,
    transcripts_dir: Path,
    model: str,
    partial_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Run one classification pass (A or B) across all video transcripts.
    Supports resumption from partial checkpoint files.
    """
    # Resume from existing partial output if available
    existing: dict[int, dict] = {}
    if partial_path.exists() and os.path.getsize(partial_path) > 0:
        try:
            df_existing = pd.read_csv(partial_path)
            for _, row in df_existing.iterrows():
                existing[int(row["ID"])] = row.to_dict()
            logger.info(
                "Pass %s: resuming — %d already classified",
                pass_name, len(existing),
            )
        except Exception:
            pass

    rows = list(existing.values())
    pending = [i for i in ids if i not in existing]

    for idx, vid_id in enumerate(tqdm(pending, desc=f"Pass {pass_name}")):
        transcript_path = transcripts_dir / f"{vid_id}.txt"
        if not transcript_path.exists():
            logger.error("Transcript missing for ID %s", vid_id)
            continue
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            logger.error("Empty transcript for ID %s", vid_id)
            continue

        prompt = prompt_template.format(transcript=transcript)
        raw = call_ollama(prompt, model, logger)
        parsed = parse_output(raw, vid_id, logger)

        row = {"ID": vid_id}
        if parsed:
            row.update(parsed)
        else:
            row.update({v: None for v in VARS})
            row["reasoning"] = ""
        rows.append(row)

        # Checkpoint save
        if (idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            logger.info(
                "Pass %s: checkpoint at %d/%d",
                pass_name, len(existing) + idx + 1, len(ids),
            )

    df = pd.DataFrame(rows)
    df.to_csv(partial_path, index=False)
    logger.info("Pass %s complete: %d rows", pass_name, len(df))
    return df


# ---------------------------------------------------------------------------
# Merge step
# ---------------------------------------------------------------------------
def merge_passes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Pass A and Pass B outputs.
    Where both passes agree, the value is accepted.
    Where they disagree, the value is set to None and the variable name
    is appended to the 'disagreement' column for manual review.
    """
    df_a = df_a.set_index("ID")
    df_b = df_b.set_index("ID")
    all_ids = df_a.index.union(df_b.index)

    rows = []
    for vid_id in all_ids:
        row = {"ID": vid_id}
        disagreements = []
        for var in VARS:
            val_a = df_a.at[vid_id, var] if vid_id in df_a.index else None
            val_b = df_b.at[vid_id, var] if vid_id in df_b.index else None

            # Coerce to int or None
            try:
                val_a = int(val_a) if val_a is not None and not pd.isna(val_a) else None
            except (ValueError, TypeError):
                val_a = None
            try:
                val_b = int(val_b) if val_b is not None and not pd.isna(val_b) else None
            except (ValueError, TypeError):
                val_b = None

            if val_a is None and val_b is None:
                row[var] = None
            elif val_a is None:
                row[var] = val_b
            elif val_b is None:
                row[var] = val_a
            elif val_a == val_b:
                row[var] = val_a
            else:
                row[var] = None  # Disagreement — requires manual review
                disagreements.append(var)

        row["disagreement"] = ",".join(disagreements) if disagreements else ""

        # Preserve reasoning from both passes
        reason_a = df_a.at[vid_id, "reasoning"] if vid_id in df_a.index and "reasoning" in df_a.columns else ""
        reason_b = df_b.at[vid_id, "reasoning"] if vid_id in df_b.index and "reasoning" in df_b.columns else ""
        row["reasoning_pass_a"] = reason_a if pd.notna(reason_a) else ""
        row["reasoning_pass_b"] = reason_b if pd.notna(reason_b) else ""

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Two-pass Moffitt populist style classifier (v3)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Ollama model tag (default: gemma3:27b)",
    )
    parser.add_argument(
        "--skip-a", action="store_true",
        help="Skip Pass A (use existing output file)",
    )
    parser.add_argument(
        "--skip-b", action="store_true",
        help="Skip Pass B (use existing output file)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    model_tag = args.model.replace(":", "-")

    log_path = RESULTS_DIR / f"classify_v3_{model_tag}_{today}.log"
    logger = setup_logging(log_path)
    logger.info("Model: %s | Date: %s", args.model, today)

    # Load video IDs from metadata
    if not METADATA_CSV.exists():
        logger.error("Metadata CSV not found: %s", METADATA_CSV)
        sys.exit(1)
    meta = pd.read_csv(METADATA_CSV)
    ids = sorted(meta["ID"].dropna().astype(int).tolist())
    logger.info("Loaded %d video IDs from %s", len(ids), METADATA_CSV)

    # Output file paths
    partial_a = RESULTS_DIR / f"pass_a_partial_{model_tag}_{today}.csv"
    partial_b = RESULTS_DIR / f"pass_b_partial_{model_tag}_{today}.csv"
    out_a     = RESULTS_DIR / f"pass_a_{model_tag}_{today}.csv"
    out_b     = RESULTS_DIR / f"pass_b_{model_tag}_{today}.csv"
    out_merge = RESULTS_DIR / f"merged_{model_tag}_{today}.csv"

    # ----- Pass A: Definition-first -----
    if not args.skip_a:
        logger.info("=== Starting Pass A (definition-first) ===")
        df_a = run_pass(
            "A", PROMPT_A_TEMPLATE, ids,
            TRANSCRIPTS_DIR, args.model, partial_a, logger,
        )
        df_a.to_csv(out_a, index=False)
        logger.info("Pass A saved → %s", out_a)
    else:
        logger.info("Skipping Pass A — loading from %s", out_a)
        df_a = pd.read_csv(out_a)

    # ----- Pass B: Decision-rule-first -----
    if not args.skip_b:
        logger.info("=== Starting Pass B (decision-rule-first) ===")
        df_b = run_pass(
            "B", PROMPT_B_TEMPLATE, ids,
            TRANSCRIPTS_DIR, args.model, partial_b, logger,
        )
        df_b.to_csv(out_b, index=False)
        logger.info("Pass B saved → %s", out_b)
    else:
        logger.info("Skipping Pass B — loading from %s", out_b)
        df_b = pd.read_csv(out_b)

    # ----- Merge -----
    logger.info("=== Merging passes ===")
    df_merged = merge_passes(df_a, df_b)
    df_merged.to_csv(out_merge, index=False)

    # ----- Summary statistics -----
    n_total = len(df_merged)
    n_disagree = (df_merged["disagreement"] != "").sum()
    logger.info(
        "Merge complete: %d videos, %d with ≥1 disagreement (%.1f%%)",
        n_total, n_disagree, 100 * n_disagree / n_total if n_total else 0,
    )

    # Per-variable agreement rates (reported in thesis Section 5.1)
    logger.info("--- Per-variable summary ---")
    for var in VARS:
        col = df_merged[var]
        n_valid = col.notna().sum()
        n_ones = (col == 1).sum()
        var_flags = df_merged["disagreement"].str.contains(var, regex=False).sum()
        agreement_rate = 100 * (1 - var_flags / n_total) if n_total else 0
        logger.info(
            "  %s: %d/%d positive (%.0f%%) | %d disagreements | "
            "inter-pass agreement: %.1f%%",
            var, n_ones, n_valid,
            100 * n_ones / n_valid if n_valid else 0,
            var_flags, agreement_rate,
        )

    print(f"\nOutputs saved to {RESULTS_DIR}/")
    print(f"  Pass A : {out_a.name}")
    print(f"  Pass B : {out_b.name}")
    print(f"  Merged : {out_merge.name}")
    print(f"  Videos requiring manual review: {n_disagree}/{n_total}")


if __name__ == "__main__":
    main()
