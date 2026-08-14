# Character Persona Replication: Layers for LLM + TTS

Primary-source research note for the memorial AI dialogue system (真白花音).  
Scope: how serious open-source / academic systems decompose character replication, and how those layers should map onto an LLM + GPT-SoVITS pipeline.  
Date: 2026-08-14. Claims below are tied to cited primary sources; unverified product internals are labeled as such.

---

## Executive summary

### Recommended layer model

| Layer (CN) | Layer (EN) | Stability | What it controls | Primary consumer |
|---|---|---|---|---|
| **人格 / 人设身份** | Persona / character identity | Session-stable (project default) | Who she is: values, red lines, self-model, memorial stance, relationship to user | **LLM system prompt** |
| **性格** | Personality / disposition | Session-stable | Stable disposition (元气、笨认真、屑、温柔、倔强) as behavioral priors | **LLM system prompt** (+ optional few-shot that *shows* traits) |
| **时期** | Era / period anchor | Project-stable default | Which public era’s voice & lore is canonical (e.g. 2020–2023 清楚元气期) | **LLM prompt + training/ref corpus filter** (not per-utterance ref index) |
| **事实** | Facts / knowledge | Retrieved per turn | Public biography, preferences, lore; what she “knows” | **LLM RAG / keyword facts** |
| **场景** | Scene / interaction context | Per turn | What this exchange is about (问候/游戏/告别…) | **LLM** (scene-conditioned few-shot + scene rules); **does not pick ref audio** |
| **语气 / 说话语气** | Speaking style / affect / prosody intent | Per utterance | How *this line* should sound (韵律/气质/劲儿) | **TTS ref-audio selection** (+ optionally LLM text affect markers) |

### Who drives what in LLM + TTS

```
[人格 + 性格 + 时期 + 红线]  ──fixed──►  LLM system prompt
[事实]                      ──RAG───►  LLM (content of lines)
[场景]                      ──match──►  LLM (few-shot + scene rules)
[说话语气]                  ──select─►  GPT-SoVITS reference audio (+ prompt text)
LLM output text             ──input──►  TTS content
```

**Rule of thumb (aligned with this repo’s `CONTEXT.md`):**

- **LLM owns** 人格 / 性格 / 时期 / 事实 / 场景 → *what is said and why*.
- **TTS ref audio owns** 说话语气 → *how it sounds*.
- **Do not** index ref audio by 场景 or 时期 for every turn; do not treat 性格 as a per-line emotion label.

This separation is not invented for this project alone. RoleLLM explicitly splits **role-specific knowledge** vs **speaking-style imitation** ([arXiv:2310.00746](https://arxiv.org/abs/2310.00746)). CharacterGLM splits **attributes** (identity, interests, experiences…) vs **behaviors** (linguistic features, emotional expressions, interaction patterns) ([arXiv:2311.16832](https://arxiv.org/abs/2311.16832)). ChatHaruhi separates **system prompt (setting/personality)** vs **retrieved script memories** vs **dialogue history** ([arXiv:2308.09597](https://arxiv.org/abs/2308.09597)). On the speech side, OpenVoice and InstructTTS treat **tone color / speaker** as separable from **style / emotion / prosody control** ([arXiv:2312.01479](https://arxiv.org/abs/2312.01479), [arXiv:2301.13662](https://arxiv.org/abs/2301.13662)); CosyVoice-instruct further separates speaker identity, speaking style (incl. emotion, rate, pitch), and fine-grained paralinguistics ([arXiv:2407.05407](https://arxiv.org/abs/2407.05407)).

---

## 1. LLM character / persona systems

### 1.1 PersonaChat — persona as profile sentences

**Primary:** Zhang et al., *Personalizing Dialogue Agents: I have a dog, do you have pets too?*  
https://arxiv.org/abs/1801.07243

- Conditioning chit-chat on a short **profile** (4–6 sentences of traits/hobbies) improves consistency and engagement.
- Layering is thin by modern standards: **persona text + dialogue history**; no separate style bank, no era lock, no multimodal path.
- Useful baseline: “personality” here is mostly **fact-like profile claims**, not prosody.

### 1.2 RoleLLM — knowledge vs speaking style

**Primary:** Wang et al., *RoleLLM*  
https://arxiv.org/abs/2310.00746  
**Code:** https://github.com/InteractiveNLP-Team/RoleLLM-public

Four-stage decomposition:

1. **Role profile construction** — descriptions, catchphrases, structured dialogues.
2. **Context-Instruct** — extract **role-specific knowledge / episodic memory**.
3. **RoleGPT** — **speaking-style imitation** via role prompting (+ retrieval).
4. **RoCIT** — role-conditioned instruction tuning.

Role profiles are not a single blob: description + catchphrases + dialogue evidence. Evaluation targets fine-grained characters, not coarse profession personas. This is the cleanest published split of **what the character knows** vs **how the character talks** in the LLM literature surveyed here.

### 1.3 CharacterGLM — attributes vs behaviors

**Primary:** Zhou et al., *CharacterGLM*  
https://arxiv.org/abs/2311.16832  
**Model/code:** https://github.com/thu-coai/CharacterGLM-6B

Character customization is configured as:

- **Attributes:** identities, interests, viewpoints, experiences, achievements, social relationships, …
- **Behaviors:** linguistic features, emotional expressions, interaction patterns, …

Success metrics: **consistency, human-likeness, engagement**. Personality is treated as a factor shaping response (e.g. gentleness vs coldness), alongside linguistic habits. Long-term memory is called out as necessary for social AI characters but is orthogonal to the attribute/behavior config.

### 1.4 Character-LLM — profile → experiences (trainable simulacra)

**Primary:** Shao et al., *Character-LLM: A Trainable Agent for Role-Playing* (EMNLP 2023)  
https://arxiv.org/abs/2310.10158  
**Code:** https://github.com/choosewhatulike/trainable-agents

- Pipeline: **profile collection → scene/experience reconstruction → SFT “experience upload”**.
- Explicit evaluation axes include **personality** (how they think/speak, tones, emotions under circumstances), **values**, **memorization**, **hallucination**, **stability**.
- **Protective experiences** reduce out-of-character modern knowledge (era/knowledge boundary as training data, not only prompt text).
- Distinguishes shallow prompt role-play from internalized experience—relevant when deciding whether a thick system prompt is enough for a memorial character.

### 1.5 ChatHaruhi — prompt + script memory retrieval

**Primary:** Li et al., *ChatHaruhi*  
https://arxiv.org/abs/2308.09597  
**Code:** https://github.com/LC1332/Chat-Haruhi-Suzumiya

Core architecture (paper Fig. 4):

1. **System prompt** — character setting / act-as instructions.
2. **Retrieved memories** \(D(q, R)\) — script excerpts relevant to query \(q\).
3. **Dialogue history** \(H\).

The paper’s own problem analysis of naive prompts:

- Fuzzy “act like X” fails to lock specific characters.
- “Know all knowledge of X” is undefined and invites hallucination.
- Style still collapses to the base LLM without examples/memories.

They separately discuss **personality consistency**, **linguistic habits** (easiest for LLMs to mimic via examples), and **story knowledge** (needs retrieval). This maps cleanly onto: system prompt (人格/性格) + RAG (事实/情节) + few-shot (语气指纹 in text).

### 1.6 Neeko — multi-character adapters

**Primary:** Yu et al., *Neeko* (EMNLP 2024)  
https://arxiv.org/abs/2402.13717  
**Code:** https://github.com/weiyifan1023/Neeko

- Dynamic **per-character LoRA** blocks + gating for multi-character role-play.
- Frames character differences as **attributes, personalities, and speaking patterns**.
- Practical lesson: if you only ever serve **one** memorial character, heavy multi-adapter machinery is optional; the conceptual split (identity block vs shared dialogue skills) still applies.

### 1.7 LiveChat — streamer personas from live video

**Primary:** Gao et al., *LiveChat* (ACL 2023)  
https://arxiv.org/abs/2306.08401  
**Code:** https://github.com/gaojingsheng/LiveChat

- 1.33M Chinese multi-party dialogues from live streams; **351 personas** with fine-grained profiles; high sessions-per-persona.
- Explicitly motivates **VTuber / virtual celebrity** downstream scenarios.
- Emphasizes that social-media chit-chat pretraining does not transfer cleanly to **live interactive** response style—relevant for VTuber-like systems where “on-stream 语气” differs from generic chat.

### 1.8 Character.AI-style systems (official product sources)

Character.AI does not publish a full academic architecture paper for persona layering. Use **official engineering/product docs** only:

- Prompt construction as runtime function of character definition, user persona, history, pinned memories, modalities, etc.:  
  https://blog.character.ai/prompt-design-at-character-ai/  
  Open-source prompt toolkit: https://github.com/character-ai/prompt-poet
- Character **Definition** as free-form identity + rules + **example dialogues** (show, don’t only list adjectives):  
  https://book.character.ai/character-guide/character-attributes/definition  
  https://support.character.ai/hc/en-us/articles/50609183646875-5-Character-Definition
- **Lorebook** as keyword-triggered world knowledge (dynamic injection without bloating every prompt):  
  https://blog.character.ai/lorebook/
- Memory product surface: https://blog.character.ai/memory/
- Inference/model family notes (not persona theory): https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/

**Observable product layering (from those sources, not reverse-engineered weights):**

| Product surface | Closest research layer |
|---|---|
| Character Definition + greeting | 人格 + 性格 + textual 语气 fingerprint |
| Example dialogues in Definition | Few-shot style |
| Lorebook / Facts | 事实 RAG |
| Pinned / story memory | Session/long-term memory |
| Optional voice | Timbre / TTS path (product feature; not a research decomposition paper) |

### 1.9 Cross-system summary: how they separate control knobs

| System | Identity / persona | Style of speech (text) | Knowledge / facts | Memory / dialogue |
|---|---|---|---|---|
| PersonaChat | Profile sentences | Implicit in profile | Profile claims | History |
| RoleLLM | Role profile | RoleGPT style imitation | Context-Instruct knowledge | Profile dialogues + generated data |
| CharacterGLM | Attributes | Behaviors (linguistic + emotion expression) | Attributes include experiences | Long-term memory called out |
| Character-LLM | Profile | Personality/tones via experiences | Experiences + protective knowledge | History; external memory optional |
| ChatHaruhi | System prompt | Linguistic habits via examples/memories | Script retrieval | History + retrieved scripts |
| Neeko | Per-character LoRA | Speaking patterns in adapter | In adapter / data | Multi-turn MCRP |
| LiveChat | Streamer profiles | Live response style in data | Persona profiles | Multi-party sessions |
| Character.AI (docs) | Definition | Example dialogues | Lorebook | History + memory features |
| **This repo (current)** | Thick system prompt (`persona.py`) | Few-shot lines + prompt “语气指纹” | Keyword facts RAG | Rolling history + compaction |

---

## 2. Personality models in dialogue agents

### 2.1 Big Five / OCEAN

**Primary training-based work:** Li et al., *BIG5-CHAT* (ACL 2025)  
https://arxiv.org/abs/2410.16491  
https://aclanthology.org/2025.acl-long.999/  
Dataset: https://huggingface.co/datasets/wenkai-li/big5_chat

- Embeds **Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism** via **human-grounded dialogues** (not only trait adjectives in prompts).
- Finds pure **prompted** trait lists weaker / less valid than training on how humans actually express traits in language.
- Psychometric inventories (BFI, IPIP-NEO) are used as evaluation tools in this literature—not as complete character definitions.

### 2.2 When trait lists help

- **Generic social agents** where the goal is controllable, measurable disposition (research bots, multi-agent sims).
- **Ablation / eval** (“make her more extraverted”) when no rich canon exists.
- As a **secondary check** that a persona is not collapsing to generic assistant agreeableness.

### 2.3 When trait lists fail for fictional / VTuber characters

Primary character papers above do **not** reduce anime/VTuber figures to OCEAN scores:

- RoleLLM argues coarse personality/profession personas lack **fine-grained role** fidelity ([arXiv:2310.00746](https://arxiv.org/abs/2310.00746)).
- ChatHaruhi stresses work-specific **personality consistency** and **linguistic habits** from scripts ([arXiv:2308.09597](https://arxiv.org/abs/2308.09597)).
- CharacterGLM uses rich **attributes + behaviors**, not a five-number vector ([arXiv:2311.16832](https://arxiv.org/abs/2311.16832)).

**Practical failure modes of OCEAN-only for 花音-class characters:**

1. **Trait collision:** “元气 + 屑 + 温柔 + 倔强 + 笨认真” is multi-label and situational; OCEAN averages wash out the comic contrast.
2. **Canon > psychometrics:** Fans recognize **catchphrases, red lines, era voice**, not BFI scores.
3. **Trait ≠ prosody:** High Extraversion does not tell GPT-SoVITS which ref clip to use.
4. **Memorial ethics:** Psych-trait role-play of a real person can overclaim “inner personality measurement”; safer to stick to **public performance persona** (what was shown on stream).

**Recommendation:** Keep **性格** as a short **character-specific trait set with do/don’t behaviors** (as in current `HUA_YIN_SYSTEM_PROMPT`), not as portable OCEAN scores. Optionally use Big Five only in offline eval rubrics if ever needed.

---

## 3. Expressive TTS + dialogue: layering style on persona

### 3.1 Three control families for “how it sounds”

| Control family | Mechanism | Examples (primary) | Pros | Cons for memorial VTuber |
|---|---|---|---|---|
| **A. Reference audio bank** | Condition synthesis on a short prompt clip (+ optional prompt text) | GPT-SoVITS ([GitHub](https://github.com/RVC-Boss/GPT-SoVITS)); CosyVoice zero-shot prompt speech ([arXiv:2407.05407](https://arxiv.org/abs/2407.05407)); GST-era style transfer ([arXiv:1803.09017](https://arxiv.org/abs/1803.09017)) | Highest **identity/timbre** lock; style is *demonstrated*, not named | Need curated clips; bad ref ⇒ bad 语气; style entangled with channel/noise |
| **B. Categorical emotion / style labels** | Happy/sad/angry or style IDs | Classical emotional TTS; CosyVoice-instruct emotion control experiments ([arXiv:2407.05407](https://arxiv.org/abs/2407.05407)) | Simple API | Labels are coarse; often **not** her categories; risk of cartoon affect |
| **C. Natural language style prompts** | Free-text style description | InstructTTS ([arXiv:2301.13662](https://arxiv.org/abs/2301.13662)); EmoVoice freestyle emotion prompts ([arXiv:2504.12867](https://arxiv.org/abs/2504.12867)); CosyVoice-instruct text instructions ([arXiv:2407.05407](https://arxiv.org/abs/2407.05407)) | Flexible, interpretable | Needs model support; hard to keep **same speaker identity** without strong cloning; prompt drift |

### 3.2 System-by-system notes

**GPT-SoVITS** — official README:  
https://github.com/RVC-Boss/GPT-SoVITS  

- Zero-shot: ~5 s vocal sample → TTS.  
- Few-shot fine-tune: ~1 min data for better similarity.  
- Inference API surface (as used in this repo’s `tts_client.py`): `ref_audio_path` + `prompt_text` / `prompt_lang` + target `text`.  
- **Implication:** multi-style control in-stack is primarily **which ref you pass**, not an OCEAN slider. Later versions advertise richer emotional expression, but control remains ref-/sampling-centric rather than a documented NL style API comparable to InstructTTS.

**CosyVoice / CosyVoice 2**  
https://arxiv.org/abs/2407.05407 · https://arxiv.org/abs/2412.10117 · https://github.com/FunAudioLLM/CosyVoice  

- Base: supervised semantic tokens + LLM + flow matching; **prompt speech** for zero-shot cloning.  
- **CosyVoice-instruct** (paper §2.4): instruction fine-tuning for **speaker characteristics**, **speaking style (emotion, gender, rate, pitch)**, and **paralinguistics** (laughter, breath, emphasis)—with examples that **omit** speaker prompt speech when training instruct control.  
- Lesson: even advanced stacks still **name three different knobs** (speaker / style / paralinguistics).

**OpenVoice**  
https://arxiv.org/abs/2312.01479 · https://github.com/myshell-ai/OpenVoice  

- Explicit design goal: clone **tone color** from reference while allowing **flexible style control** (emotion, accent, rhythm, pauses, intonation) **not constrained** to the reference clip’s style.  
- Strong conceptual precedent for: **fixed identity embedding** + **variable style**.

**StyleTTS 2**  
https://arxiv.org/abs/2306.07691 · https://github.com/yl4579/StyleTTS2  

- Models **style as latent** via diffusion; can sample style **without** reference speech for naturalness.  
- Useful for read-speech naturalness; for **memorial identity**, reference-free style sampling is usually the wrong default (identity drift risk).

**InstructTTS**  
https://arxiv.org/abs/2301.13662  

- Frames two prior control modes: (1) **categorical style index**, (2) **reference speech**—and proposes (3) **natural language style prompt**.  
- Notes ref-speech styles are often **not intuitive/interpretable**; NL prompts improve interpretability but require annotated data and disentanglement (style vs speaker vs content via MI minimization in their method).

**EmoVoice**  
https://arxiv.org/abs/2504.12867  

- LLM-based TTS with **freestyle text emotion prompts**; releases EmoVoice-DB with NL emotion descriptions.  
- Emotion control is **utterance-level affect**, not character personality.

**Global Style Tokens (background terminology)**  
Wang et al., https://arxiv.org/abs/1803.09017  

- Unsupervised **style tokens** controlling speed/speaking style independent of text content; style transfer from a reference clip.  
- Historical root of “style embedding” language in modern TTS.

### 3.3 Dialogue systems vs TTS: the missing joint layer

LLM persona papers almost never specify **which acoustic style** accompanies a line. TTS papers almost never specify **character red lines**. A multimodal memorial system must own the **crosswalk**:

```
LLM produces text (and optionally a 语气 tag)
        │
        ▼
语气 tag ──select──► ref audio from bank (same speaker, same 时期 filter)
        │
        ▼
GPT-SoVITS(text, ref) ──► waveform
```

Without that crosswalk, you get **人格正确、声线飘** or **声线稳、这句劲儿不对**—both fail this repo’s **多模态在场感** acceptance unit (`CONTEXT.md`).

---

## 4. Terminology: speaking style vs personality vs emotion

Literature is inconsistent; “style” is overloaded. Recommended **ubiquitous language** for this project (speech + dialogue):

| Prefer (EN) | Prefer (CN) | Meaning | Not the same as |
|---|---|---|---|
| **Persona / character identity** | **人格 / 人设身份** | Who she is; values; memorial stance; boundaries | Per-line mood |
| **Personality / disposition** | **性格** | Stable behavioral priors | Acoustic emotion label |
| **Speaking style (text)** | **文风 / 语气指纹（文本）** | Lexical habits, length, catchphrases, taboo formats | Prosody |
| **Speaking style (speech) / prosodic manner** | **说话语气** | How this utterance should *sound* | Topic/scene |
| **Emotion / affect (utterance)** | Prefer **说话语气** in product copy; if needed academically: **句级情感/affect** | Short-term affective coloring of one line | 性格 |
| **Prosody** | **韵律** | Pitch, duration, energy, pauses (acoustic realization) | Persona text |
| **Timbre / tone color / speaker identity** | **声线 / 音色** | Who’s voice | Style of delivery |
| **Scene / context** | **场景** | Interaction situation / topic class | Ref-audio key |
| **Era / period** | **时期** | Canon time anchor for lore + default voice | Per-line tag |
| **Facts / knowledge** | **事实** | Retrievable public information | Style |

### Why avoid bare “情绪 / emotion” as the product axis

1. **Psychology vs acoustics:** “情绪” suggests internal state; memorial systems should not claim to simulate private affect of a real person.
2. **TTS literature** uses emotion as **categorical or prompted acoustic control** (EmoVoice, CosyVoice-instruct)—useful internally, misleading in UX.
3. **This repo already standardized** on **说话语气** as the ref-audio driver (`CONTEXT.md`).

### Mapping common paper phrases → project terms

| Paper phrase | Map to |
|---|---|
| Role profile / character definition | 人格 (+ partial 性格) |
| Personality traits (Big Five) | 性格 (only if generic; weak for VTuber canon) |
| Speaking style imitation (RoleLLM) | 文风 / 语气指纹（文本） |
| Linguistic features / habits | 文风 |
| Emotional expressions (CharacterGLM behaviors) | Split: text affect vs 说话语气 |
| Style tokens / style embedding | 说话语气（声学） |
| Tone color (OpenVoice) | 声线 |
| Prompt speech / ref audio | 参考音 (implements 说话语气 + 声线) |
| Instruction / NL style prompt | Alternative control for 说话语气 (if engine supports) |

---

## 5. Implications for 花音

### 5.1 What you already have (keep)

Current stack already matches the better half of the literature:

| Component | Code / asset | Layer |
|---|---|---|
| Thick system prompt (era lock, traits, 语气指纹, scene rules, red lines) | `dialogue/persona.py` | 人格 + 性格 + 时期 + textual 语气 + 场景规则 |
| Keyword fact RAG | `dialogue/persona_context.py` + `docs/persona/facts-kb.json` | 事实 |
| Scene-matched few-shot lines | `docs/persona/fewshot-lines.json` | 文风 / 场景-conditioned text style |
| Rolling memory + compaction | `dialogue/memory.py`, `conversation.py` | Session memory |
| Single fixed GPT-SoVITS ref | `dialogue/tts_client.py` | 声线 + default 说话语气 |

This is essentially **ChatHaruhi-style** (prompt + retrieval + history) with a **CharacterGLM-like** attribute/behavior prompt, plus a **single-style** TTS path.

### 5.2 What to add for multi-style ref selection (without conflating layers)

**Add a fourth runtime path: 语气 → ref bank**, independent of fact RAG and scene few-shot.

#### A. Build a **说话语气** ref bank (not a 场景 bank)

1. From high-quality clips (e.g. `data/dataset_precision/`), curate **short, clean, single-speaker** refs (~3–10 s), all within **default 时期** (清楚元气期) unless user explicitly enters another era mode.
2. Label refs by **听感 / 说话语气**, e.g.:
   - `元气轻快` `温柔低声` `屑笑` `认真短促` `惊讶短促` `告别收束` `害羞小声` …
3. Store: `path`, `prompt_text`, `prompt_lang`, `style_tags[]`, `period`, `quality_score`, **never** only a scene name.

Scene may be *correlated* with style (告别 → often 温柔), but the **index key is 说话语气**. Same scene can need different 劲儿 (“游戏翻车” vs “游戏认真”).

#### B. Predict 语气 from **assistant text** (and optional user text), not from scene alone

Recommended pipeline:

```
user text ──► scene match ──► facts + few-shot ──► LLM ──► reply text
                                                              │
                                                              ▼
                                                    语气 classifier / rules
                                                    (on reply text ± user)
                                                              │
                                                              ▼
                                                    pick ref(style*) from bank
                                                              │
                                                              ▼
                                                    GPT-SoVITS(reply, ref)
```

- Prefer tagging the **model’s own line** (what will be spoken). Scene-only selection recreates the anti-pattern forbidden in `CONTEXT.md`.
- Start with **deterministic rules + keyword/lexicon** on reply text (感叹号密度、口癖、告别词、自嘲标记); add a small classifier only if rules plateau.
- Optional: have the LLM emit a **machine-only** style tag in a side channel (not spoken)—but keep it a closed enum of 说话语气, not free “emotion essay”.

#### C. Keep personality out of the ref picker

| Do | Don’t |
|---|---|
| Ref picker sees 语气 tags + quality + period | Ref picker sees “性格=屑” as a permanent default clip |
| 性格 shapes **wording** in the system prompt | 性格 forces one acoustic stereotype every turn |
| Default ref = neutral 元气 (period-locked) | Switch ref by topic keywords alone |
| Fall back to default ref on low confidence | Randomize refs (identity flicker) |

性格 is already multi-trait simultaneous in the prompt (“五特质同时在场”). Acoustic style should **modulate within** that identity, not swap identities.

#### D. Align text few-shot with speech ref (optional but high value)

When scene match retrieves few-shot lines, prefer examples whose **source clip** (if known) is in the same 语气 cluster as the selected ref—or at least does not contradict it. This is the multimodal version of RoleLLM’s joint knowledge+style construction.

#### E. Engine alternatives (only if ref banks are insufficient)

- Stay on **GPT-SoVITS multi-ref** first (matches current training investment).
- Consider **OpenVoice-like** tone-color / style split only if you need style orthogonal to available refs ([arXiv:2312.01479](https://arxiv.org/abs/2312.01479)).
- NL style prompts (InstructTTS / EmoVoice / CosyVoice-instruct) are research-grade options; they do **not** replace a curated 花音 ref bank for identity lock without re-validation.

#### F. Memorial / safety constraints on style expansion

- Do not add refs that break **时期** default or red lines (e.g. grief-performance voice as default entertainment style).
- Do not label styles as clinical emotions of the real person.
- Keep **AI 复刻** disclosure independent of acoustic liveliness.

### 5.3 Concrete architecture sketch

```
prepare_chat_messages():
  system = 人格+性格+时期+红线+文风规则     # fixed
  + facts(user)                            # 事实
  + fewshot(scene(user))                   # 场景 → 文风 examples
  + history

reply = LLM(...)

style = infer_speaking_style(reply, user)  # 说话语气 only
ref   = select_ref(style, period=default)  # bank; fallback default
wav   = GPT-SoVITS(reply, ref)
```

Acceptance check (from `CONTEXT.md` **多模态在场感**): content + 语气意图 + 声线 cohere in one turn.

---

## 6. Uncertainties / do-not-overclaim

1. **No public end-to-end paper** fully specifies “VTuber memorial = RoleLLM layers + GPT-SoVITS multi-ref.” Joint recommendations above are **synthesis** from primary sources, not a single validated standard.
2. **Character.AI** persona internals beyond published blogs/docs are unknown; do not claim hidden architecture details.
3. **GPT-SoVITS** multi-style quality depends heavily on ref cleanliness and fine-tune data; primary README claims few-shot cloning capability, not guaranteed 语气 taxonomy accuracy for a specific idol corpus.
4. **Big Five** validity for **fictionalized public personas** of real streamers is ethically and scientifically weak; BIG5-CHAT validates trait expression in general dialogue, not VTuber canon fidelity.
5. **Emotion labels** in TTS papers are often acted or crowdsourced categories; they may not match fan-perceived 花音 劲儿.
6. **NL style prompts** that work on CosyVoice-instruct / EmoVoice are **not** drop-in for the current GPT-SoVITS deployment without engine change and re-eval.
7. **LiveChat** supports the VTuber dialogue domain shift claim; it does not provide a ready 花音 model.
8. **Protective experiences** (Character-LLM) reduce hallucination for historical figures; for a modern VTuber memorial, the analogue is **red lines + “no pretended private memory”**—already in the system prompt—not full experience-SFT unless you invest in that training path.
9. Listener studies for **this** project’s multi-ref policy are not yet primary evidence; treat §5 as engineering guidance pending A/B listening tests.
10. Avoid claiming the system **is** 真白花音 or recovers private personality; literature on role-play is about **simulacra / imitation quality**, not ontological identity.

---

## 7. Chinese-friendly terminology suggestions (for `CONTEXT.md` glossary)

Suggested entries to add or align (short definitions; keep existing **Avoid** discipline):

| Term | Definition sketch | Avoid |
|---|---|---|
| **人格 / 人设身份** | 稳定的“她是谁”：纪念立场、价值观、红线、自我边界；写入系统提示，不每句切换。 | 人格模拟（单独、含义不清）, 本人 |
| **性格** | 稳定 Disposition：元气/笨认真/屑/温柔/倔强等同时在场的行为先验；约束**台词选择**，不直接索引参考音。 | OCEAN 分数当人设, 每句性格标签 |
| **说话语气** | （已有）这句合成语音听起来的劲儿；**参考音选择主轴**。 | 情绪, 情感, mood（作产品主轴时） |
| **文风 / 语气指纹（文本）** | 句长、口癖、禁列表腔、标点习惯等**文本**层风格；由系统提示 + few-shot 负责。 | 与说话语气混用 |
| **场景** | （已有）互动情境/话题类；约束台词与事实召回，**不**直接选参考音。 | 场景=参考音标签 |
| **时期** | （已有）人设与声线年代锚点；默认清楚元气期。 | 每句时期标签 |
| **事实** | 可检索的公开设定与偏好；RAG 注入，禁止编造隐私记忆。 | 假装亲历 |
| **声线 / 音色** | 说话人身份的声学底色；由模型 + 参考音共同锁定。 | 用声线指代说话语气 |
| **参考音** | GPT-SoVITS 的 ref clip + prompt text；实现声线锁定与说话语气。 | 参考音=场景 |
| **多模态在场感** | （已有）台词内容、语气意图、合成声线同一轮一致。 | 只像文风, 只像音色 |

**One-line stack slogan for docs:**  
**人格定边界，性格定习惯，场景定话题，事实定内容，说话语气定参考音。**

---

## 8. Primary source index

### LLM / dialogue / persona

| Work | URL |
|---|---|
| PersonaChat (Zhang et al.) | https://arxiv.org/abs/1801.07243 |
| RoleLLM (Wang et al.) | https://arxiv.org/abs/2310.00746 |
| CharacterGLM (Zhou et al.) | https://arxiv.org/abs/2311.16832 |
| Character-LLM (Shao et al.) | https://arxiv.org/abs/2310.10158 |
| ChatHaruhi (Li et al.) | https://arxiv.org/abs/2308.09597 |
| Neeko (Yu et al.) | https://arxiv.org/abs/2402.13717 |
| LiveChat (Gao et al.) | https://arxiv.org/abs/2306.08401 |
| BIG5-CHAT (Li et al.) | https://arxiv.org/abs/2410.16491 |
| Character.AI Prompt Design | https://blog.character.ai/prompt-design-at-character-ai/ |
| Character.AI Definition guide | https://book.character.ai/character-guide/character-attributes/definition |
| Character.AI Lorebook | https://blog.character.ai/lorebook/ |
| Prompt Poet | https://github.com/character-ai/prompt-poet |

### TTS / style / emotion

| Work | URL |
|---|---|
| GPT-SoVITS (official repo) | https://github.com/RVC-Boss/GPT-SoVITS |
| CosyVoice | https://arxiv.org/abs/2407.05407 |
| CosyVoice 2 | https://arxiv.org/abs/2412.10117 |
| OpenVoice | https://arxiv.org/abs/2312.01479 |
| StyleTTS 2 | https://arxiv.org/abs/2306.07691 |
| InstructTTS | https://arxiv.org/abs/2301.13662 |
| EmoVoice | https://arxiv.org/abs/2504.12867 |
| Global Style Tokens | https://arxiv.org/abs/1803.09017 |

### Related VTuber / AI streamer context (secondary to architecture)

| Work | URL | Note |
|---|---|---|
| LiveChat | https://arxiv.org/abs/2306.08401 | Primary for live/VTuber dialogue data motivation |
| AI VTuber fandom study | https://arxiv.org/abs/2509.10427 | Social/HCI; not a layer architecture paper |
| Viewer perception of AI VTubers | https://arxiv.org/abs/2509.20817 | Perception study; not control-stack design |

---

*End of research note. Implementation of multi-ref selection should be tracked separately (ADR + listening eval), not implied as already shipped.*
