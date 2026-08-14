# Speaking-style inventory and control for reference-audio TTS

Research notes for the AI 真白花音 project: which closed-set **说话语气** inventory and control architecture fit zero-shot / few-shot reference-conditioned TTS (especially GPT-SoVITS) when only ~1k high-precision Chinese clips are available.

**Primary-source rule:** claims below cite arXiv, ACL/IEEE venues, official GitHub READMEs/model docs, or original dataset papers. Blog recaps are not used as evidence.

---

## Executive summary

For this project—fine-tuned GPT-SoVITS with **single-reference inference** and ~1.2k clean Chinese clips—the most appropriate design is a **small closed set of speaking styles (说话语气, ~5–7 labels)**, each backed by a **human-curated multi-ref bank** of short clips, selected **per utterance** by an **LLM self-tag** in a constrained schema (not by scene, period, or free-form emotion prose). GPT-SoVITS, VALL-E-style codec LMs, F5-TTS, and CosyVoice zero-shot ICL all treat **reference/prompt speech** as the primary carrier of prosody and affect; discrete emotion labels or natural-language style prompts only help when the backbone was trained for them (CosyVoice-Instruct, InstructTTS, OpenVoice base+SSML, IndexTTS2 style prompt). This stack does not have a native emotion-ID or instruct path, so **style control = which `ref_audio_path` (and matching `prompt_text`) is sent to `/tts`**. Scene should stay on the dialogue/content side only—matching both this repo’s domain language and the research pattern that separates *what is said* from *how it sounds*. Prefer a character-centric inventory (e.g. 元气 default / 温柔 / 俏皮 / 惊讶 / 低落 / 认真) over full Ekman-6 or large acted SER taxonomies; those datasets explain *why* papers pick N classes, not which N a VTuber memorial voice should ship.

---

## 1. Categorical vs dimensional emotion for expressive TTS

### 1.1 Claim: classic speech corpora use small categorical sets, often Ekman-adjacent + neutral

| Resource | N classes | Categories (as published) | Primary source |
|---|---|---|---|
| Ekman basic emotions (theory) | 6 | anger, disgust, fear, happiness, sadness, surprise | [Ekman universal emotions overview](https://www.paulekman.com/universal-emotions/) |
| Berlin EmoDB | 7 | anger, boredom, disgust, fear, happiness, sadness, neutral | Burkhardt et al., Interspeech 2005, [ISCA archive](https://www.isca-archive.org/interspeech_2005/burkhardt05b_interspeech.html) |
| ESD (EN+ZH parallel) | 5 | neutral, happy, angry, sad, surprise | Zhou et al., [arXiv:2105.14762](https://arxiv.org/abs/2105.14762); [GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| RAVDESS speech | 8 | neutral, calm, happy, sad, angry, fearful, disgust, surprised | Livingstone & Russo, *PLoS ONE* 2018, [doi:10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391) |
| IEMOCAP categorical | ~9 (+ other) | anger, happiness, excitement, sadness, frustration, fear, surprise, disgust, neutral | Busso et al., *Lang. Resour. Eval.* 2008, [doi:10.1007/s10579-008-9076-6](https://doi.org/10.1007/s10579-008-9076-6) |
| MER 2023 (Chinese multimodal) | 6 discrete + valence | neutral, anger, happiness, sadness, worry, surprise (+ continuous valence) | Lian et al., [arXiv:2304.08981](https://arxiv.org/abs/2304.08981) |
| EmoBox cross-corpus eval set | 4 | angry, happy, neutral, sad | Ma et al., [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) |

**Why N is small in papers**

- **Actability and perception:** EmoDB and RAVDESS use few acted categories so performers and listeners share a label set; EmoDB reports high recognition rates after filtering (e.g. anger ~97%, disgust ~80% in the 2005 paper).
- **Parallel content control:** ESD deliberately holds text fixed across 5 emotions for conversion/synthesis experiments ([arXiv:2105.14762](https://arxiv.org/abs/2105.14762)).
- **Annotation reliability:** IEMOCAP provides both categorical labels and continuous valence/activation/dominance; many SER papers collapse to 4 classes (angry/happy/sad/neutral, often merging happiness+excitement) because agreement is only moderate on fine labels ([Busso 2008](https://doi.org/10.1007/s10579-008-9076-6); EmoBox IEMOCAP map, [arXiv:2406.07162](https://arxiv.org/abs/2406.07162)).
- **Cross-corpus comparability:** EmoBox standardizes many datasets but still uses a **4-class** balanced cross-corpus test—evidence that large N does not travel well across corpora ([arXiv:2406.07162](https://arxiv.org/abs/2406.07162)).

### 1.2 Claim: dimensional (PAD / valence–arousal) is better for continuous nuance, but needs different conditioning machinery

- Zhou et al. argue categorical labels under-cover the emotional spectrum and propose **Pleasure–Arousal–Dominance (PAD)** control for LM-based TTS, mapping categories into PAD via psychology-grounded anchors while **not requiring emotion labels at TTS training time** ([arXiv:2409.16681](https://arxiv.org/abs/2409.16681)).
- IEMOCAP already annotated continuous valence, activation, dominance with SAM scales ([Busso 2008](https://doi.org/10.1007/s10579-008-9076-6)).
- MER 2023 tracks include both discrete labels and dimensional valence ([arXiv:2304.08981](https://arxiv.org/abs/2304.08981)).

**Implication:** dimensional control is research-attractive for breadth, but it assumes a model that accepts continuous style vectors. GPT-SoVITS inference does not expose PAD knobs; style lives in the reference waveform.

### 1.3 Claim: “emotion” in SER taxonomies ≠ “speaking style” in dialogue TTS

- InstructTTS distinguishes **pre-defined categorical styles** (limited diversity) from **reference speech** (uninterpretable) and proposes **natural language style prompts** ([arXiv:2301.13662](https://arxiv.org/abs/2301.13662)).
- Spoken-LLM / StyleTalk models dialogue-relevant **speaking style** as `<emotion, speed, volume>` with dialogue-oriented emotion words (neutral, cheerful, sad, friendly, unfriendly), not full Ekman ([arXiv:2402.12786](https://arxiv.org/abs/2402.12786)).
- This repo’s domain language already prefers **说话语气** (how it sounds) over 情绪/mood as the ref-selection axis (`CONTEXT.md`).

---

## 2. How major zero-shot / ref-conditioned systems control style

### 2.1 Architecture map

| System | Primary style/emotion channel | Secondary controls | Official / paper |
|---|---|---|---|
| **VALL-E** | ~3 s **acoustic prompt** (codec tokens); preserves prompt emotion & acoustic environment | Phoneme/text content prompt | [arXiv:2301.02111](https://arxiv.org/abs/2301.02111) |
| **GPT-SoVITS** | **Reference audio** (+ prompt text/lang); few-shot finetune improves speaker fidelity | Sampling (`top_k/p`, temperature), `speed_factor`; optional multi-aux refs in API | [GitHub RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) |
| **F5-TTS** | **Audio prompt** mel + ref transcript; style/prosody copied via speech infilling / ICL | Duration via length ratio / sway sampling | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885); [GitHub SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) |
| **StyleTTS 2** | Latent **style diffusion** from text (no ref); or **style encoder on reference** for multi-speaker / transfer | Fixed-length style vector \(s\), AdaIN | [arXiv:2306.07691](https://arxiv.org/abs/2306.07691); [GitHub yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) |
| **CosyVoice / 2** | Zero-shot **ICL reference speech** for timbre/prosody/emotion; **instruct** path for NL + fine tags | CosyVoice 2: ~1500 h instructed data; emotions e.g. Happy/Sad/Surprised/Angry/Fearful/Disgusted/Calm/Serious; `[laughter]`, rate, dialect | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407), [arXiv:2412.10117](https://arxiv.org/abs/2412.10117); [GitHub FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| **OpenVoice** | **Decouple**: base TTS controls style (emotion, accent, rhythm…); **tone-color converter** clones speaker from short ref | SSML / base-speaker style APIs | [arXiv:2312.01479](https://arxiv.org/abs/2312.01479); [GitHub myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) |
| **InstructTTS** | **Natural language style prompt** → discrete latent expressive TTS | Not primarily ref-audio cloning | [arXiv:2301.13662](https://arxiv.org/abs/2301.13662) |
| **IndexTTS2** | **Timbre prompt** + optional **separate emotion/style prompt** (or NL soft instruction); explicit disentanglement | Duration / token-count control | [arXiv:2506.21619](https://arxiv.org/abs/2506.21619) |
| **EmoVoice** | LLM backbone + **freestyle NL emotion prompts** | Phoneme-boost variant for content | [arXiv:2504.12867](https://arxiv.org/abs/2504.12867) |

### 2.2 Claims with evidence

**Claim A — Reference audio is the default style bus for zero-shot codec / flow TTS.**  
VALL-E states it can preserve the speaker’s emotion and acoustic environment of the acoustic prompt ([arXiv:2301.02111](https://arxiv.org/abs/2301.02111)). F5-TTS conditions generation on reference mel + text and is trained as text-guided speech infilling, so speaker and style come from the unmasked audio context ([arXiv:2410.06885](https://arxiv.org/abs/2410.06885)). CosyVoice frames zero-shot cloning as in-context learning over reference speech tokens ([arXiv:2407.05407](https://arxiv.org/abs/2407.05407); CosyVoice 2 upgrades instructed control while keeping ICL, [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)).

**Claim B — GPT-SoVITS is few-shot/zero-shot ref-driven; it is not an emotion-label TTS.**  
Official features: “Zero-shot TTS: Input a 5-second vocal sample…” and “Few-shot TTS: Fine-tune … with just 1 minute of training data” ([GPT-SoVITS README](https://github.com/RVC-Boss/GPT-SoVITS)). This project’s client posts `ref_audio_path`, `prompt_text`, `prompt_lang` to `/tts` (`dialogue/tts_client.py`)—the same contract as the official API. Style change without architecture change means **changing the reference clip**.

**Claim C — Systems that want style *independent* of the clone ref explicitly disentangle channels.**  
OpenVoice separates tone color (from ref) from style (from base TTS) ([arXiv:2312.01479](https://arxiv.org/abs/2312.01479)). IndexTTS2 takes a timbre prompt and an optional different emotion/style prompt ([arXiv:2506.21619](https://arxiv.org/abs/2506.21619)). StyleTTS 2 samples a latent style (with or without ref encoding) ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)).  
**Without that disentanglement, using one clip for both identity and affect is the correct mental model** (GPT-SoVITS / F5 / VALL-E default).

**Claim D — Text/instruct emotion control requires trained instruct capacity, not just an LLM tag.**  
CosyVoice 2 integrates ~1500 h of instructed data and lists discrete emotion names plus fine-grained tags ([arXiv:2412.10117](https://arxiv.org/abs/2412.10117)). InstructTTS and EmoVoice are built around NL style/emotion prompts ([arXiv:2301.13662](https://arxiv.org/abs/2301.13662), [arXiv:2504.12867](https://arxiv.org/abs/2504.12867)). Passing “sad” as plain synthesis text into GPT-SoVITS does **not** implement these mechanisms.

---

## 3. LLM → TTS control pipelines for dialogue

### 3.1 Common patterns (from primary systems/papers)

| Pattern | How it works | Example primary source | Fit for GPT-SoVITS? |
|---|---|---|---|
| **A. LLM emits closed style tag + text** | Response includes enum label; TTS conditioned on tag or mapped resource | Spoken-LLM: predict style tags then Azure expressive TTS ([arXiv:2402.12786](https://arxiv.org/abs/2402.12786)); EmoNews: sentiment → emotion tag → PromptTTS ([arXiv:2506.13894](https://arxiv.org/abs/2506.13894)) | **Yes** if tag maps to ref bank |
| **B. Separate classifier after LLM text** | Generate text → SER/sentiment model → emotion ID → TTS | EmoNews: distilled RoBERTa over pure LLM tagging (LLM over-predicted some emotions) ([arXiv:2506.13894](https://arxiv.org/abs/2506.13894)) | Possible offline; online adds latency/error |
| **C. Inline SSML / vocal-burst tags** | `<speak>`, `[laughter]`, emphasis spans | OpenVoice uses SSML-capable base TTS ([arXiv:2312.01479](https://arxiv.org/abs/2312.01479)); CosyVoice 2 fine tags ([arXiv:2412.10117](https://arxiv.org/abs/2412.10117)) | **No** native SSML in GPT-SoVITS; tags would be spoken literally unless stripped |
| **D. Latent / continuous style** | PAD vector, style diffusion sample, learned embedding | StyleTTS 2 ([arXiv:2306.07691](https://arxiv.org/abs/2306.07691)); PAD LM-TTS ([arXiv:2409.16681](https://arxiv.org/abs/2409.16681)) | **No** without engine change |
| **E. Multi-prompt disentangled ref** | Timbre ref + emotion ref | IndexTTS2 ([arXiv:2506.21619](https://arxiv.org/abs/2506.21619)) | Not GPT-SoVITS default; single fused ref is the practical substitute |

### 3.2 Claim: dialogue papers treat style as a small structured side channel, not free prose

- StyleTalk styles are structured triples; Azure TTS realizes them ([arXiv:2402.12786](https://arxiv.org/abs/2402.12786)).
- EmoNews uses **five** emotions for news SDS: neutral, happy, sad, angry, surprised—then PromptTTS ([arXiv:2506.13894](https://arxiv.org/abs/2506.13894)). That set matches ESD’s five ([arXiv:2105.14762](https://arxiv.org/abs/2105.14762)).
- EmoNews also reports that **prompt-only LLM emotion labeling was biased**, motivating a dedicated analyzer—relevant if we rely on LLM self-tags without constraints or calibration ([arXiv:2506.13894](https://arxiv.org/abs/2506.13894)).

### 3.3 Claim: content-style consistency is a product metric, not automatic

- Spoken-LLM’s premise: same text spoken differently should elicit different responses; style is first-class in dialogue ([arXiv:2402.12786](https://arxiv.org/abs/2402.12786)).
- This project’s acceptance unit is 多模态在场感: line content, 语气意图, and synthesized voice align in one turn (`CONTEXT.md`). Pipeline-wise that is **tag → ref selection → TTS**, not text-only LLM + fixed ref.

---

## 4. Practical guidance for character voice clone with ~1k clean clips

### 4.1 Claim: inventory size should track (a) audible clusters in *this* speaker’s data and (b) reliable LLM/human labeling—not SER leaderboard N

Evidence-backed heuristics:

1. **Cross-corpus SER often collapses to 4 classes** when labels must be comparable (EmoBox, [arXiv:2406.07162](https://arxiv.org/abs/2406.07162)).
2. **Emotional VC/TTS benchmarks that need Chinese parallel data use 5 (ESD)** ([arXiv:2105.14762](https://arxiv.org/abs/2105.14762)).
3. **Dialogue SDS examples use ~5 emotion tags** (EmoNews, [arXiv:2506.13894](https://arxiv.org/abs/2506.13894)) or a compact style triple (Spoken-LLM, [arXiv:2402.12786](https://arxiv.org/abs/2402.12786)).
4. **GPT-SoVITS few-shot is marketed around minutes of data for one voice**, not multi-style supervised emotion heads ([README](https://github.com/RVC-Boss/GPT-SoVITS)). With ~1179 precision clips, budget styles so each has **enough distinct, clean exemplars** for a ref bank (practically: multiple candidates per style, 1–3 production refs after listening).

**Recommended range for this project: 5–7 closed 说话语气 labels**, including a high-prior default (元气/轻快). Going to 10+ (full RAVDESS/IEMOCAP) usually yields empty or confusable bins on spontaneous VTuber speech and unstable LLM tagging.

### 4.2 Claim: separate **scene** from **speaking style / affect channel**

- Research systems that care about style either (i) condition TTS on style tags/refs, or (ii) disentangle timbre vs emotion prompts (IndexTTS2, OpenVoice)—they do not index style by topic.
- This repo already states: **场景** constrains lines/facts and must **not** select reference audio; **说话语气** drives ref choice; **时期** is a persona/voice anchor, not per-utterance (`CONTEXT.md`).

### 4.3 Claim: multi-ref banks are the practical control surface for single-ref engines

- VALL-E / F5 / CosyVoice ICL / GPT-SoVITS all improve or change delivery by changing the prompt clip.
- IndexTTS2’s dual-prompt design is the “ideal” research form of multi-ref (timbre bank × emotion bank) ([arXiv:2506.21619](https://arxiv.org/abs/2506.21619)). For GPT-SoVITS, approximate with **one bank of style-labeled clips of the same speaker**, each clip carrying *both* identity and 语气.
- Optional GPT-SoVITS `aux_ref_audio_paths` (API) is for tone fusion, not a documented per-emotion controller; treat multi-ref **selection** (pick one primary ref) as the main design, not fusion, until measured.

### 4.4 Operational recipe (engine-agnostic, GPT-SoVITS-native)

1. **Define closed enum** of 说话语气 (see §5).
2. **Offline:** human (or human+tool) assign candidate clips from `dataset_precision` to styles; keep only clips that *sound* like the target 语气 under headphones.
3. **Ship:** 1 default + 1–3 alternates per style (fallback if file missing).
4. **Online:** LLM outputs `{ "style": <enum>, "text": "..." }` (or style field beside streamed sentences).
5. **Map** `style → ref_audio_path + prompt_text + prompt_lang`.
6. **Synthesize** via existing API fields (already in `TTSClient`).
7. **Evaluate** by listening for 在场感 (content ↔ 语气 ↔ voice), not SER accuracy.

---

## 5. Chinese / Japanese-relevant resources

| Resource | Relevance | Source |
|---|---|---|
| **ESD Chinese half** | 10 ZH speakers × 5 emotions, parallel text; standard for emotional VC/TTS | [arXiv:2105.14762](https://arxiv.org/abs/2105.14762); [GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| **AISHELL-3** | Large multi-speaker Mandarin TTS; **emotion-neutral** (~85 h, 218 speakers)—good for multi-speaker TTS, not emotion taxonomy | [arXiv:2010.11567](https://arxiv.org/abs/2010.11567) |
| **MER 2023** | Chinese multimodal; 6 discrete + valence; more “in-the-wild” than acted ESD | [arXiv:2304.08981](https://arxiv.org/abs/2304.08981) |
| **CosyVoice / 2** | Strong ZH (and JA in later Fun-CosyVoice) zero-shot + instruct emotions listed in Chinese/English | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117); [CosyVoice README](https://github.com/FunAudioLLM/CosyVoice) |
| **F5-TTS** | Public multilingual training includes Chinese; zero-shot ref style | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) |
| **GPT-SoVITS** | Official cross-lingual list includes Chinese and Japanese | [README](https://github.com/RVC-Boss/GPT-SoVITS) |
| **EmoBox** | Multilingual SER toolkit; shows how ZH and other corpora map into shared names | [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) |

**Note:** AISHELL-3 being emotion-neutral is a reminder that “Chinese TTS dataset” ≠ “expressive style inventory.” For 花音, the inventory must come from **her** clips’ audible variation, with ESD/MER only as *naming priors*, not training labels we must match.

Japanese: CosyVoice family and GPT-SoVITS list Japanese support; a full JA acted emotion corpus survey is out of scope here and not required for a Chinese-primary memorial voice. Do not invent JA taxonomy claims without a specific corpus paper.

---

## 6. Implications for 花音 project

### 6.1 Recommended control architecture

```
User utterance
    → LLM (persona + scene/memory for *content only*)
    → constrained fields: speech_text + 说话语气 ∈ closed set
    → style → ref bank lookup (wav + prompt transcript)
    → GPT-SoVITS /tts (single ref)
    → playback
```

- **Do not** index refs by 场景 or 时期.
- **Do not** rely on embedding emotion words in the spoken text.
- **Do not** wait for GPT-SoVITS native emotion IDs; research alternatives (CosyVoice-Instruct, IndexTTS2, OpenVoice) are engine migrations, not drop-in flags.

### 6.2 Recommended closed-set options

**Option A — Character 说话语气 (preferred, 6 labels)**  
Aligned with memorial VTuber dialogue and `CONTEXT.md` wording:

| ID | 说话语气 | Listening intent (for ref curation) | Rough SER prior (optional) |
|---|---|---|---|
| `genki` | 元气/轻快 | Default bright energy, clear smile in voice | happy / neutral-high |
| `soft` | 温柔 | Lower intensity, warm, unhurried | calm / soft-happy |
| `tease` | 俏皮/调侃 | Playful bite, slightly faster or singsong | happy / friendly |
| `surprise` | 惊讶 | Raised pitch range, short lifted contour | surprise |
| `down` | 低落 | Softened energy, not theatrical crying | sad (mild) |
| `serious` | 认真/沉稳 | Steady, less bounce | neutral / serious |

**Option B — ESD-compatible 5 (portable, more “emotion” flavored)**  
`neutral | happy | angry | sad | surprise` per [ESD](https://arxiv.org/abs/2105.14762) / [EmoNews](https://arxiv.org/abs/2506.13894).  
**Risk:** `angry` may be rare or out-of-character; `neutral` is ambiguous for a naturally energetic VTuber (default should be `genki`, not flat neutral).

**Option C — Minimal 4 (most robust labeling)**  
`bright | soft | down | surprise` — matches EmoBox’s “keep it small” lesson ([arXiv:2406.07162](https://arxiv.org/abs/2406.07162)). Use if Option A bins are too sparse after listening.

**Start with Option A; collapse to Option C if listening tests show empty/confusable bins.**

### 6.3 Who decides the label?

| Decider | Role | When |
|---|---|---|
| **LLM self-tag (primary)** | Emit `说话语气` from closed enum in the same generation step as dialogue text (JSON/schema / tool field) | Online every utterance; mirrors Spoken-LLM “predict style then TTS” ([arXiv:2402.12786](https://arxiv.org/abs/2402.12786)) |
| **Light rules (secondary)** | Bias/prior: e.g. force `genki` default; map obvious markers; never invent new labels | Safety net when model omits tag or streams partial text |
| **Offline human for ref bank (required)** | Choose which wavs represent each 语气 | One-time / periodic curation of multi-ref bank |
| **Classifier (optional, not primary online)** | emotion2vec / SER to *suggest* clusters for the bank; EmoNews-style text sentiment if LLM tags are biased ([arXiv:2506.13894](https://arxiv.org/abs/2506.13894); emotion2vec used in EmoBox/IndexTTS2 eval) | Offline labeling aid; A/B if self-tags drift |

**Not recommended as primary:** free-form NL emotion strings into GPT-SoVITS; training a 6-way SER head on 1k unlabeled-in-emotion live clips and trusting it in the hot path without listening validation.

### 6.4 Concrete next engineering steps (out of scope for this note, listed for traceability)

1. Freeze enum in config (IDs + Chinese display names + default `genki`).
2. Curate ref bank under e.g. `model/refs/<style>/`.
3. Extend `TTSClient.synthesize` to accept `ref_audio_path` / prompt overrides per call.
4. Teach persona prompt: output style enum; strip any style tokens from spoken text.
5. Listening eval matrix: same sentence × all styles; plus dialogue scenarios with scene held constant while 语气 changes.

---

## 7. Uncertainties and what we should NOT overclaim

1. **No peer-reviewed GPT-SoVITS paper** was used here; behavior claims come from the **official GitHub README/API** and this repo’s integration. Community guides are not treated as primary evidence.
2. **We did not measure** how many of the 1179 precision clips fall into each audible 语气—inventory size is a research-informed recommendation, not a data audit.
3. **SER accuracy ≠ dialogue naturalness.** High ESD classification accuracy does not mean a memorial chat should speak in acted “Angry.”
4. **Dimensional PAD** ([arXiv:2409.16681](https://arxiv.org/abs/2409.16681)) is not implementable on stock GPT-SoVITS without a different backbone or a learned ref-retrieval over continuous embeddings (untested here).
5. **Disentangled dual-ref** (IndexTTS2, OpenVoice) is better *in principle* for “same timbre, different emotion from another speaker’s clip”; with a **single-speaker memorial corpus**, same-speaker style refs are the honest approach—we should not claim full timbre–emotion independence.
6. **LLM self-tags can be biased** (EmoNews observation, [arXiv:2506.13894](https://arxiv.org/abs/2506.13894)); constrained enums + default prior + listening tests are mitigations, not proofs of calibrated affect.
7. **CosyVoice instruct emotion lists** describe *that* model’s training distribution, not labels we can feed to GPT-SoVITS.
8. **Japanese taxonomies** were not deeply surveyed; do not claim a JA standard set for this Chinese-primary system.
9. **“Emotionally aware SDS” literature is still thin** relative to TTS papers; EmoNews and Spoken-LLM are useful patterns, not industry standards.
10. **Multi-ref fusion** (`aux_ref_audio_paths`) is not validated in this project; do not equate API presence with a recommended emotion architecture.

---

## 8. Source list (verified primary)

1. Wang et al., *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* (VALL-E) — https://arxiv.org/abs/2301.02111  
2. Du et al., *CosyVoice* — https://arxiv.org/abs/2407.05407  
3. Du et al., *CosyVoice 2* — https://arxiv.org/abs/2412.10117  
4. Chen et al., *F5-TTS* — https://arxiv.org/abs/2410.06885  
5. Li et al., *StyleTTS 2* — https://arxiv.org/abs/2306.07691  
6. Qin et al., *OpenVoice* — https://arxiv.org/abs/2312.01479  
7. Yang et al., *InstructTTS* — https://arxiv.org/abs/2301.13662  
8. Zhou et al., *Emotional Voice Conversion: Theory, Databases and ESD* — https://arxiv.org/abs/2105.14762  
9. ESD GitHub — https://github.com/HLTSingapore/Emotional-Speech-Data  
10. Lin et al., *Spoken-LLM / StyleTalk* (ACL 2024) — https://arxiv.org/abs/2402.12786  
11. Matsuura et al., *EmoNews* — https://arxiv.org/abs/2506.13894  
12. Ma et al., *EmoBox* — https://arxiv.org/abs/2406.07162  
13. Zhou et al., *Emotional Dimension Control in LM-Based TTS* — https://arxiv.org/abs/2409.16681  
14. Deng et al., *IndexTTS2* — https://arxiv.org/abs/2506.21619  
15. Zhao et al., *EmoVoice* — https://arxiv.org/abs/2504.12867  
16. Shi et al., *AISHELL-3* — https://arxiv.org/abs/2010.11567  
17. Lian et al., *MER 2023* — https://arxiv.org/abs/2304.08981  
18. Busso et al., *IEMOCAP* — https://doi.org/10.1007/s10579-008-9076-6  
19. Livingstone & Russo, *RAVDESS* — https://doi.org/10.1371/journal.pone.0196391  
20. Burkhardt et al., *Berlin EmoDB* — https://www.isca-archive.org/interspeech_2005/burkhardt05b_interspeech.html  
21. GPT-SoVITS official repo — https://github.com/RVC-Boss/GPT-SoVITS  
22. CosyVoice official repo — https://github.com/FunAudioLLM/CosyVoice  
23. F5-TTS official repo — https://github.com/SWivid/F5-TTS  
24. StyleTTS2 official repo — https://github.com/yl4579/StyleTTS2  
25. OpenVoice official repo — https://github.com/myshell-ai/OpenVoice  
26. Project domain language — `/home/mtr/tt/tts/CONTEXT.md`  
27. Project TTS client contract — `/home/mtr/tt/tts/dialogue/tts_client.py`  
28. Precision dataset scale (~1179 ZH clips) — `/home/mtr/tt/tts/data/dataset_precision/README.md`

---

*Document purpose: research input for choosing 说话语气 inventory and ref-bank control for AI 真白花音. Not a training recipe and not a claim that any single paper “solved” memorial dialogue TTS.*
