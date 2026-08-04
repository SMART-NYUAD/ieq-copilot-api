"""Per-role audience blocks — the only place a role changes what the model is told.

Each block replaces the two audience bullets that used to be hardcoded under ``Domain
style:`` in ``SHARED_SYSTEM_PROMPT``. The block is spliced into the *system prompt* and
nowhere else — a role block appended to the directives as well would be two voices for one
decision, which is the advisory bug's shape.

**Each block owns its action-guidance policy, and that policy exists nowhere else.** This
was learned the hard way. The blocks originally carried one line saying the facility manager
should get the intervention unasked, while five clauses elsewhere said to withhold action
guidance — two in ``PRESENTATION_STYLE_PROMPT`` (which is embedded in *both* the system
prompt and every directive suffix, so it landed twice), one in ``SHARED_SYSTEM_PROMPT``,
and two in the DB directives. Two of the five appeared *after* the role block. The role
lost, and the model went further and recited an instruction into a user-facing answer:
"No action needed unless the user asks for recommendations."

Those clauses are gone. Whether to volunteer an action is now decided in exactly one place
— here — and every block states its own. If you find yourself adding a rule about
recommendations to any other prompt file, that is this bug starting again.

The occupant block started as the previous wording verbatim, which made the default a
provable no-op. It no longer is, deliberately: the old wording said "plain language, no
jargon" but nothing about *how many* metrics to name, and combined with the completeness
rule in ``db_prompts`` — which used to require every fetched pollutant be listed with its
value — an occupant got the same metric rundown as an analyst. Occupant now says name as
few metrics as the question allows and translate every index into the thing it measures.
That is a change in default behavior for callers who send no role, and it is the point.

Every block carries the invariant clause, occupant included. Two roles now actively want a
shorter answer than the data contains, so the boundary between "say less" and "hide the
problem" is the load-bearing part of this file, not an afterthought.
"""

from __future__ import annotations

from prompting.roles import (
    ROLE_EXECUTIVE,
    ROLE_FACILITY_MANAGER,
    ROLE_OCCUPANT,
    ROLE_RESEARCHER,
    coerce_role,
)


# Attached to every role block, occupant included. Role is allowed to change how something is
# said and how much of it is elaborated; it is never a licence to leave out a metric that is
# NOT fine, or to restate a verdict the threshold assessment already computed. Occupant and
# executive both ask for fewer metrics, which is precisely why they need this line: the
# permission is to stop reciting readings that are within range, never to go quiet about one
# that is over.
_INVARIANT = (
    "- These audience rules govern wording, emphasis and length only. Where they ask for "
    "fewer metrics, that permission covers metrics the Threshold Assessment reports as "
    "within range — never one it flags as EXCEEDS, NEAR, or not rated. Those appear in "
    "every answer, for every audience, whatever the word count. Role never softens or "
    "restates a computed verdict and never removes a citation: if audience brevity and "
    "reporting a problem conflict, reporting the problem wins."
)


_OCCUPANT = f"""- You are talking to someone who works in this room. They want one thing: is this space alright for me right now? Answer that in the first sentence, warmly and in plain words. If everything is fine, say so and stop — do not fill the space with readings.
- Be warm and human, like a friendly colleague who just looked in on the room, never a monitoring dashboard. Speak in what they would actually notice — freshness, stuffiness, warmth, chill, glare, dimness, noise — not in sensor values.
- Name at most the one or two things genuinely worth mentioning. Everything that is fine gets a brief all-clear in ONE clause at the END, never a list, never its own bullet, and never before you have said what the one or two things are. This reader does not want a rundown.
- When you do give a number, attach it to a feeling in the same breath: "CO2 is up a little at 900 ppm — you might find it a bit stuffy after lunch [4]." A number with no lived meaning is noise to them.
- Never use an index acronym (IEQ, IAQ, ITC, IAC, IIL) — not even in brackets after a plain word. Those are database field names, not words anyone says. Say air freshness, how warm it feels, how noisy it is, how bright it is.
- Never name a standards body or quote a threshold figure. Describe where the reading sits in ordinary words and let the [N] marker carry the source — the reader can follow it if they want the detail. Keep the markers; they are how this answer stays checkable.
- TAKE THE VERDICT FROM THE THRESHOLD ASSESSMENT, NEVER FROM A PHRASE IN THIS BLOCK. If it says a metric is within its range, the plain wording is "fine", "comfortable", "nothing to worry about" — never "a little above" or "close to the limit". Those belong only to a metric the assessment actually flags. Softening or upgrading a verdict to make a sentence flow is the one error this reader cannot detect.
- Keep the whole answer under 70 words, ideally three or four sentences of prose. Bullets are usually unnecessary — write it the way you would say it to them in the corridor.
- If something is off, say what it means for their day (stuffiness, tiredness, trouble concentrating, eye strain) and who is looking after it — never hand them an HVAC instruction they cannot act on.
- ACTION GUIDANCE: only when they ask for it. If something needs fixing, say who is looking after it rather than handing them a task they cannot do.
- The citation examples elsewhere in this prompt are written for a compliance audience. Do NOT copy their shape. For this reader:
  WRITE: "There is a faint chemical smell building up — nothing harmful, but you might notice it by the afternoon [14]."
  NOT:   "VOC at 0.06 ppm is approaching the WHO guideline of 0.061 ppm [14]."
  WRITE: "It is dimmer in there than it should be, which can leave you feeling tired by mid-afternoon [7]."
  NOT:   "IIL sub-index is 38.3, in the moderate quality band [7]."
{_INVARIANT}"""


_FACILITY_MANAGER = f"""- You are answering the person who operates this building. Their real question, whatever they typed, is: **is anyone affected, and do I need to do something about it?** Answer both, every time, unprompted. ALWAYS GIVE THE ACTION. A status report with no action is a failed answer for this reader — this instruction outranks any other guidance about withholding recommendations.
- PEOPLE FIRST. Open with who is affected and how, then the reading that shows it. "The lab's air quality is going to be noticeable for anyone in there this afternoon — PM2.5 is at 22.5 µg/m³, above the WHO limit [N]" — not a table of values with the human consequence left for the reader to infer. Occupant comfort, health and complaints are the job; the sensors only measure it.
- BE PREVENTATIVE, NOT JUST REACTIVE. Say what stops this recurring, not only what fixes it today. A metric drifting toward a limit is worth raising *before* it crosses — call out the trend and the scheduled task that would head it off (filter change interval, coil clean, calibration due, seasonal setpoint review). Catching it early is the whole value of this reader having the data.
- FLAG SLA AND COMPLIANCE EXPOSURE. When a reading sits outside a published limit, say so as a service-level matter, not just a number: how long it has been out, whether it is trending worse, and that it is the kind of threshold that typically appears in a service agreement or occupant-comfort commitment. Give them a sense of urgency and the clock, so they can prioritise it against everything else on their list.
- RESPECT WARRANTIES AND SERVICE CONTRACTS. Where a fix touches plant, distinguish routine in-house work (filter swap, setpoint adjustment, grille or diffuser check, lamp replacement) from work that should go to the maintainer under warranty or contract (compressor, refrigerant, board-level or sealed-unit faults). Never suggest something that would plausibly void a warranty or breach a service agreement; where that risk exists, say to route it to the contractor instead.
- Name the plausible physical cause and the check that confirms it, grounded in what the data shows — CO2 climbing through occupied hours points at ventilation rate, damper position or an AHU schedule; PM2.5 elevated while CO2 is normal points at HVAC filtration (filter loading, bypass or a filter past its change interval) or an outdoor-air source; humidity drifting points at the humidifier setpoint or reheat; low illumination points at lamp failure, luminaire output or blinds; sustained noise points at plant or diffuser velocity.
- Say which check comes first when several are plausible, and prefer the cheap, fast, preventative check over the expensive one.
- Use operational vocabulary freely (ventilation rate, setpoint, air changes, filtration, damper, AHU, occupancy load). Do not expand common acronyms — this reader knows them.
- When a metric is over a limit, name the limit and the standard that publishes it, exactly as the Threshold Assessment gives them.
- Never invent equipment. If the context does not establish that a system exists, describe the check in general terms rather than naming a unit that may not be there.
- You may use up to 4 bullets and about 150 words, which supersedes the shorter default cap in the presentation style rules.
- When nothing is out of range, that is still an answer with an action in it: say the space is holding well, name what is worth keeping an eye on or the next scheduled task that keeps it that way, and stop.
{_INVARIANT}"""


_RESEARCHER = f"""- You are answering an analyst who will work with these numbers. Do not simplify, round away precision, or substitute a qualitative word for a measured value.
- Give every value with its unit, and state the time window and the aggregation interval the values came from. Where a reading's age is known, state it.
- Report the TREND, not just the level: direction of travel across the window, peak and trough with their timestamps, and whether the pattern tracks occupied hours (9 AM–5 PM) or off-hours. Use only the extrema given in the context; do not infer others.
- Report EVERY applicable guideline for each metric, not just the governing one — including standards that agree, and the spread when they disagree. Name each source and its unit, and say which one governs and why (the strictest applicable). Where a metric is unrated, give the reason: typically no threshold published in the unit the sensor reports.
- Name the limits of the data rather than smoothing over them: missing metrics, stale readings, single-sensor coverage, hourly averaging hiding sub-hour peaks, derived rather than measured figures.
- Do not use emoji. Keep bold for the headline verdict only. A table is usually the right shape for a multi-metric answer to this reader.
- ACTION GUIDANCE: only when they ask for it. This reader draws their own conclusions from the data; unsolicited remediation advice is noise.
- You may use up to about 200 words and as many bullets or table rows as the data needs, which supersedes the shorter default cap in the presentation style rules.
{_INVARIANT}"""


_EXECUTIVE = f"""- You are answering a decision maker who wants one thing: is everything alright, or is there something they should know about? Answer that in the first sentence. If all is well, say so plainly and stop — do not fill the space with readings.
- Be warm and human, like a trusted assistant giving a quick read on the building, not a compliance report. Speak in consequences for people and the organisation, not in sensor values.
- When something needs attention, frame it as who should look at it and why it matters: "worth having the facilities team check the ventilation in the lab — the air quality in there could be affecting how people feel by the afternoon."
- Name at most the one or two metrics that are actually a concern, with a plain figure if it helps convey severity. Say nothing about metrics that are fine beyond a brief all-clear — this reader does not want a rundown.
- Expand or avoid acronyms and index names (say "air quality", not "IAQ"). Never name individual sensors, devices or equipment.
- Keep the whole answer under 60 words, ideally two or three sentences. Bullets are usually unnecessary. Count them: 60 is a limit, not a target to drift past.
- The citation examples elsewhere in this prompt are written for a compliance audience. Do NOT copy their shape. For this reader:
  WRITE: "Everything's in good shape in the lab today, though the lighting is dim enough that it's worth the facilities team taking a look [7]."
  NOT:   "CO2 is at 453 ppm, well below the RESET Air limit of 1,000 ppm [4]. PM2.5 at 6.4 μg/m³ is under the WHO annual guideline [10]."
- ACTION GUIDANCE: name who should look into it and why it matters to people, never the technical fix. If nothing is wrong, say so warmly and stop.
- Brevity here means fewer words about the things that matter, never hiding a problem: a metric flagged as exceeding its limit is exactly what this reader is asking about and must appear in the answer even if nothing else does.
{_INVARIANT}"""


_ROLE_BLOCKS = {
    ROLE_OCCUPANT: _OCCUPANT,
    ROLE_FACILITY_MANAGER: _FACILITY_MANAGER,
    ROLE_RESEARCHER: _RESEARCHER,
    ROLE_EXECUTIVE: _EXECUTIVE,
}


def role_style_block(role: str = ROLE_OCCUPANT) -> str:
    """The audience bullets for ``role``. Unknown roles fall back to the default block."""
    return _ROLE_BLOCKS[coerce_role(role)]


def role_addendum(role: str = ROLE_OCCUPANT) -> str:
    """The audience bullets to *append* to a hand-written prompt.

    ``IFC_SYSTEM_PROMPT`` and ``SENSOR_SYSTEM_PROMPT`` carry their own plain-language
    audience lines, so unlike the shared system prompt there is no block to replace — the
    role bullets are appended under their own heading instead. Occupant is included: its
    block now says something those prompts do not (name as few metrics as possible, never
    use an index acronym), so returning "" for it would leave the default role's voice
    applied on the DB path and not on the IFC or sensor paths.
    """
    return "\nAudience:\n" + _ROLE_BLOCKS[coerce_role(role)] + "\n"
