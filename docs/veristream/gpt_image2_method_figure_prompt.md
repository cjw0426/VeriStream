# GPT Image 2 Prompt: VeriStream Method Figure

本文件用于生成论文双栏宽度的 VeriStream 方法总览图。推荐先使用 Primary Prompt 生成，再根据实际
错误使用 Revision Prompt 定向修正。图内文字全部使用英文。

## Primary Prompt

```text
Use case: scientific-educational
Asset type: publication-ready method overview figure for a top-tier computer vision or machine learning paper
Target venue style: CVPR / ICCV / NeurIPS / ICLR
Aspect ratio: very wide landscape, approximately 2.4:1, designed to span two paper columns

Primary request:
Create a rigorous, elegant scientific system diagram for a training-free causal long-video question-answering agent named VeriStream. The central idea is not dense frame sampling. VeriStream keeps exact recent visual perception intact, organizes the long history as query-independent evidence memory, and lets a bounded LLM controller autonomously acquire additional evidence through validated tools.

Do not put a large decorative title inside the figure. Use a small top-left label only:
"VeriStream: Autonomous Causal Evidence Retrieval"

Use a clean white background, flat vector-like graphics, crisp arrows, compact horizontal English labels, and generous whitespace. The figure must remain readable when reduced to a two-column paper width. Do not invent any component, training stage, or unrestricted video-seeking capability.

GLOBAL LAYOUT
Use five left-to-right columns with two visually separated horizontal lanes:

1. Causal Video Input
2. Persistent Evidence Memory
3. Question-conditioned Planning and Retrieval
4. Bounded Autonomous Tool Loop
5. Grounded Final Answer

The upper lane is the historical evidence path. The lower lane is a continuous blue "Exact Recent-4" current-visual path. The lower lane must bypass historical indexing and join only at final reasoning. Use a vertical dashed red causal boundary at the question time. No future frame may cross this boundary.

Use one consistent miniature video example throughout: a person picks up a red cup, places it on a table, and later leaves. Thumbnails should be simple, small, and subordinate to the scientific diagram.

────────────────────────────────────────────────────────
COLUMN 1 — CAUSAL VIDEO INPUT
────────────────────────────────────────────────────────

Draw an ordered video timeline ending at a vertical dashed red line labeled exactly:
"Causal boundary"

Split the accessible prefix into:
- upper branch: "Historical frames"
- lower branch: exactly four blue-outlined thumbnails labeled "Exact Recent-4"

Add a small separation lock and the exact note:
"Recent-4 excluded from history index"

Historical frames flow to L0 memory. Exact Recent-4 continues as an uninterrupted blue arrow along the bottom toward Column 5. Show a few faded future frames to the right of the causal boundary with a prohibition mark and the label "inaccessible future"; no arrow may originate from them.

────────────────────────────────────────────────────────
COLUMN 2 — PERSISTENT EVIDENCE MEMORY
────────────────────────────────────────────────────────

Place this entire column inside a subtle pale-green background region labeled:
"Query-independent, video-isolated index"

At the left of this region, show:
"L0 RawFrameMemory"
with the compact subtitle:
"raw frames · timestamps · CLIP embeddings"

From L0, create two parallel proposal branches for each 12-second block:

A. a teal branch labeled:
"Temporal coverage"
with the subtitle:
"4 uniform frames / 12 s"

B. a coral branch labeled:
"Change proposals"
with three tiny aligned traces labeled:
"CLIP spans 1 / 2 / 4"
followed by a compact normalization box labeled:
"per-span robust MAD normalization"
then a peak selector labeled:
"local peaks + temporal NMS"
and the output label:
"≤ 2 before/after proposals"

Make the logical role explicit with a small coral annotation:
"CLIP proposes where, not what happened"

Merge and deduplicate coverage frames, proposal pairs, and minimal local context into one small frame strip labeled:
"Block pack · ≤ 8 frames"

Send this pack into one coral model box labeled:
"Frozen VLM Perception"
with the subtitle:
"one block call"

The perception output must split into:

1. "Block overview"
2. "Proposal ID → paired assessment"

Immediately after the paired assessment, draw a hierarchical gate with exactly three outcomes:

- green: "local semantic event → admit"
- gray: "global visual change → boundary metadata"
- red outline: "no meaningful change → reject"

Under the gate, add a tiny structural validation side loop:
"missing / invalid ID → one targeted repair"

The repair loop must return only to structural validation. It must not imply semantic correction, training, or repeated unconstrained retries.

Show the searchable L1 memory as exactly two node types:

- teal node: "OverviewMemory"
  subtitle: "interval summary + L0 provenance"

- coral node: "TransitionMemory"
  subtitle: "before → change → after"

Only admitted local semantic events create searchable TransitionMemory. Global changes remain non-searchable boundary metadata, and rejected proposals remain audit records; neither may appear as searchable TransitionMemory nodes.

Draw thin graph edges:
- temporal links between adjacent OverviewMemory nodes
- parent/child links between OverviewMemory and TransitionMemory
- predecessor/successor links between adjacent TransitionMemory nodes

Draw thin dashed provenance arrows from both L1 node types back to L0 frames.

Add the exact note at the bottom of this region:
"Persistent memory: L0 + L1 only"

Do not show persistent L2, answer memory, cross-video memory, or model-written experience memory.

────────────────────────────────────────────────────────
COLUMN 3 — QUESTION-CONDITIONED PLANNING AND RETRIEVAL
────────────────────────────────────────────────────────

At the top, show:
"Question + options"
entering a blue-gray box labeled:
"EvidenceNeed Planner"

The planner must output small structured chips, not prose paragraphs. Show:
- "N1 · historical event"
- "N2 · attribute"
- for temporal order, two atomic event chips "N3 · event A" and "N4 · event B" connected by "R1: before(N3, N4)"

Add a thin rule strip below the planner:
"Preserve tense and temporal semantics"

Show a visible bypass from the planner to the lower Exact Recent-4 lane labeled:
"High-confidence current-only fast path"

For historical or both-scope needs, show a compact L1 retrieval module with three parallel branches:

- teal: "BGE semantic"
- gray: "BM25 lexical"
- coral: "Conditional CLIP"
  subtitle: "attribute · spatial · state only"

The three branches must first enter a strict gate labeled exactly:
"Absolute relevance gate · may return empty"

Only admitted candidates then enter:
"Weighted RRF"

Follow this with:
"Complementary selection"

Under Complementary selection, show only this compact formula:
"utility = need coverage gain − redundancy penalty"

Show three tiny redundancy cues beneath the formula:
"text" · "visual" · "time overlap"

Retrieval searches L1 memory nodes. It does not directly search an arbitrary number of raw frames. Its output enters a dashed container labeled:
"Working Evidence · temporary per question"

The Working Evidence container is not persistent memory. Use a pale lavender dashed outline to distinguish it from L0/L1.

────────────────────────────────────────────────────────
COLUMN 4 — BOUNDED AUTONOMOUS TOOL LOOP
────────────────────────────────────────────────────────

This is the conceptual center of the figure. Clearly separate model decision from deterministic execution.

At the top, draw a coral box labeled:
"LLM Tool Controller"

Inside or immediately beneath it, use the compact subtitle:
"observe needs + evidence + tool result + remaining budget"

The LLM Tool Controller chooses exactly one next action from a horizontal tool row:

- "Search L1"
- "Inspect L0"
- "Expand graph"
- "Compare nodes"
- "Finish"

Use restrained line icons: magnifying glass, eye, linked arrows, comparison brackets, and check mark. The labels are more important than the icons.

Below the tool row, draw a neutral gray box labeled:
"Validated Tool Executor"

Its subtitle must be:
"causal boundary · ID validity · state transition · hard budgets"

The control flow must form a clear closed loop:

LLM Tool Controller
→ one selected tool
→ Validated Tool Executor
→ tool result updates Working Evidence
→ updated evidence returns to LLM Tool Controller

Use one prominent curved feedback arrow labeled:
"reassess evidence sufficiency"

Finish exits the loop toward final reasoning. Budget exhaustion exits with unresolved needs; it must not be drawn as verified evidence.

Enforce these tool semantics visually:

- Search L1 locates candidate memory for an unresolved Need.
- Inspect L0 is allowed only from a localized L1 memory ID and follows its provenance frames.
- Expand graph follows only existing before / after / parent / child links.
- Compare nodes operates only on two already localized memories.
- Finish is chosen when important needs are sufficiently covered or no useful action remains.

Add the exact boundary note:
"No arbitrary timestamps or frame IDs"

Show a compact evidence-state strip:
"unlocated → located → text-admitted / visually verified / relation verified"

Place "partial" and "rejected" as side exits that do not reach final evidence.

For temporal-order reasoning, show a tiny inset:
two separately localized TransitionMemory nodes
→ "verify endpoints"
→ "compare peak times"
→ relation edge

Do not imply that two top-ranked candidates automatically prove temporal order.

Place five small muted-gold budget badges in one tidy row:

"≤ 4 executed tools"
"≤ 2 search rounds"
"≤ 2 graph expansions"
"≤ 2 visual actions"
"≤ 8 history frames"

Add a tiny note:
"Finish is free"

The hard budgets constrain actions and visual cost, not the number of final evidence cards. Do not draw a fixed evidence-count limit.

────────────────────────────────────────────────────────
COLUMN 5 — GROUNDED FINAL ANSWER
────────────────────────────────────────────────────────

Merge exactly two inputs:

1. the lower blue lane labeled "Exact Recent-4 raw frames"
2. admitted rows from Working Evidence labeled "grounded historical evidence"

Feed both into a blue model box labeled:
"Frozen VLM Reasoning"

Output a compact box labeled:
"Answer"

Place this decision rule beside the merge:
"current state → Recent-4"
"past events → grounded evidence"

Do not show unverified candidates, rejected memories, future frames, or raw retrieval rankings entering final reasoning.

────────────────────────────────────────────────────────
VISUAL DESIGN SYSTEM
────────────────────────────────────────────────────────

Use a restrained, colorblind-conscious academic palette:

- charcoal text and outlines: #263238
- current lane and reasoning: blue #356AE6
- coverage and OverviewMemory: teal #2A9D8F
- proposals and TransitionMemory: coral #E76F51
- tool budgets and small emphasis: muted gold #D6A434
- Working Evidence: pale lavender #EEE8F7 with #7A6F9B outline
- neutral executor and validation: cool gray #E9EEF2
- rejected or inaccessible: restrained red #C84B4B
- background: pure white #FFFFFF

Use colors by semantic role, not as decoration. No gradients, neon colors, dark background, glossy effects, 3D rendering, heavy shadows, decorative blobs, or photorealistic scenery.

Use one modern sans-serif typeface. Keep all text horizontal. Stage headings should be compact and bold; body labels should be short. Use small-radius rectangles, thin 1.5–2 px arrows, consistent arrowheads, aligned baselines, and generous spacing. Avoid cards nested inside cards. Avoid arrows crossing unrelated components.

The visual hierarchy should be immediately readable at three levels:

1. blue current lane versus historical evidence lane
2. offline persistent memory versus question-time autonomous retrieval
3. LLM decision versus validated tool execution

────────────────────────────────────────────────────────
NON-NEGOTIABLE LOGICAL CONSTRAINTS
────────────────────────────────────────────────────────

- Training-free; all models are frozen.
- Only frames at or before the causal boundary are accessible.
- Exact Recent-4 never enters the history index.
- CLIP change distance proposes candidate locations and never declares an action or state change.
- Search operates over L1 OverviewMemory and admitted TransitionMemory.
- Inspect accesses L0 only through provenance of an already localized memory ID.
- Expand follows only explicit graph links.
- Compare uses already localized memories.
- The LLM autonomously chooses the next tool and when to Finish.
- The deterministic executor validates and executes calls; it does not choose the next tool.
- A visual Need enters final evidence only after Inspect or Compare verification.
- Temporal order requires two independently localized endpoints and an explicit relation check.
- Working Evidence is temporary and cleared after each question.
- Persistent memory is isolated by video ID and contains no answer-written experience.
- Retrieval may return empty.
- No fixed maximum number of evidence cards is shown.
- No persistent L2 or L3 memory is shown.
- No training loop, fine-tuning, gradients, external web tools, arbitrary seeking, or unrestricted timestamp search.

TEXT ACCURACY AND COMPOSITION CONSTRAINTS

Render every required label in clear English. Do not add pseudo-words, lorem ipsum, unexplained acronyms, extra equations, or invented model names. If space is limited, remove decorative icons and optional thumbnail detail before removing logical arrows or required labels. Keep text inside its container, prevent text-arrow overlap, and ensure every arrow has an unambiguous source and target.
```

## Revision Prompt

首次生成后，先用下方 Checklist 找出错误，再将具体问题填写到 `[OBSERVED DEFECTS]`。不要仅说“更清晰”，
而要指出错误标签或错误连线。

```text
Revise the existing VeriStream method figure. Preserve the correct five-column composition, white background, palette, thumbnail example, and all already-correct labels. Fix only the defects listed below.

[OBSERVED DEFECTS]
- ...

Mandatory logical corrections:

1. Preserve the strict two-lane design. Exact Recent-4 is a continuous blue bypass and never enters historical indexing.
2. Preserve the causal boundary. Future frames are inaccessible and have no outgoing data-flow arrow.
3. Keep persistent memory at L0 and L1 only. L1 contains OverviewMemory and admitted TransitionMemory; Working Evidence is temporary.
4. Keep CLIP as a location proposal mechanism only. Frozen VLM Perception performs the local/global/none semantic assessment.
5. Keep the hierarchical gate outcomes distinct: local event is searchable, global visual change is boundary metadata only, and no meaningful change is rejected.
6. Keep retrieval order exact: BGE / BM25 / conditional CLIP → absolute relevance gate that may return empty → Weighted RRF → complementary selection.
7. Replace any deterministic or automatic tool-selection policy with the correct closed loop:
   LLM Tool Controller → one selected tool → Validated Tool Executor → Working Evidence update → reassess evidence sufficiency.
8. The deterministic executor validates causal boundaries, memory IDs, states, and budgets. It does not choose Search, Inspect, Expand, Compare, or Finish.
9. Search targets L1. Inspect reaches L0 only from an already localized memory. Expand follows graph links. Compare uses two localized memories.
10. Temporal order uses two independently localized and verified endpoints plus peak-time comparison.
11. Partial, rejected, unresolved, and budget-exhausted evidence must not enter final reasoning.
12. Final reasoning receives only grounded historical evidence and Exact Recent-4 raw frames.
13. Do not show a fixed evidence-card count. Keep only action and visual-cost budgets.

Typography corrections:
- Correct every malformed or pseudo-English label.
- Keep all text horizontal and inside its container.
- Remove label-arrow overlap and unrelated arrow crossings.
- Prefer fewer decorative details over smaller text.
- Keep the figure readable at two-column paper width.

Do not add training, fine-tuning, future-frame access, arbitrary timestamps, persistent L2/L3 memory, answer-written memory, external tools, or extra model modules. Return one polished wide scientific figure, not a slide, poster, dashboard, or marketing infographic.
```

## Final Typography Pass

若整体逻辑正确但文字仍有乱码，使用这一轮只修文字，不允许重画结构。

```text
Perform a typography-only cleanup of the existing figure. Preserve every module, arrow, position, color, thumbnail, and logical relationship. Correct malformed English, clipped text, inconsistent capitalization, and label overlap. Use the exact labels already specified in the source prompt. Do not add, remove, merge, or reconnect any module. Keep all text horizontal and readable at two-column paper width.
```

## Generation Checklist

生成后逐项检查：

- [ ] Recent-4 是独立蓝色直通通道，且没有进入 L0/L1。
- [ ] causal boundary 右侧帧不可访问。
- [ ] CLIP 只生成 proposal，没有直接输出动作语义。
- [ ] local/global/none 三类 gate 结果去向不同。
- [ ] 搜索顺序为 absolute gate → RRF → complementary selection。
- [ ] 图中明确区分 LLM Tool Controller 与 Validated Tool Executor。
- [ ] 箭头构成 Controller → Tool → Executor → Evidence → Controller 的闭环。
- [ ] Inspect 只能从已定位 L1 节点回看 L0。
- [ ] Temporal order 使用两个独立端点和显式关系边。
- [ ] Working Evidence 是临时结构，不是 L2。
- [ ] final reasoning 只接收 grounded evidence 与 Recent-4。
- [ ] 没有固定 evidence 数量上限。
- [ ] 没有训练、未来帧、任意时间戳搜索或跨视频记忆。
- [ ] 所有文字清晰，无乱码、遮挡和交叉箭头。

## Suggested GPT Image 2 Settings

```text
model: gpt-image-2
quality: high
orientation: landscape
aspect ratio: 2.4:1 or the widest available landscape option
format: PNG
background: white
```

建议将最终图片保存为：

```text
VeriStream.png
```

然后在论文或 `实验报告.md` 中以双栏宽图插入，并另行撰写 caption，不要依赖图内大标题解释方法。

## Tailored Revision Prompt for the Current `VeriStream.png`

上传当前 `VeriStream.png` 作为 edit target，然后直接使用以下英文提示词。该版本针对当前图片中已经观察
到的结构、比例和文字问题，不要求模型自行猜测缺陷。

```text
Use case: infographic-diagram
Asset type: final two-column method figure for a top-tier CVPR / ICCV / NeurIPS paper
Input image: the attached VeriStream.png is the edit target. Preserve its core five-stage left-to-right narrative and its correct causal logic, but redraw the full figure cleanly rather than patching isolated pixels.

PRIMARY GOAL
Transform the existing figure into a compact, publication-ready scientific method diagram. The result must look like a paper figure, not a slide, dashboard, poster, or product infographic. Use a very wide landscape canvas, approximately 2.4:1, ideally 2400 x 1000 or the widest available equivalent. Remove the large in-figure title entirely because the paper caption supplies the title. Increase effective font size by reducing nonessential text and vertical stacking. The complete diagram must remain legible when printed at approximately 180 mm width.

PRESERVE THESE CORRECT IDEAS
- Left-to-right causal flow.
- A red causal boundary with inaccessible future frames and no future-frame arrows.
- A continuous blue Exact Recent-4 lane that bypasses historical indexing and joins only at final reasoning.
- Query-independent, video-isolated persistent L0/L1 memory.
- CLIP change proposals followed by frozen VLM semantic assessment.
- Question-conditioned retrieval, temporary Working Evidence, an autonomous LLM tool loop, and grounded final reasoning.
- The restrained blue / teal / coral / lavender / cool-gray palette on a pure white background.

REDRAW AS FIVE COMPACT STAGES
1. Causal Video Prefix
2. Persistent L0/L1 Memory
3. EvidenceNeed Retrieval
4. Bounded Tool Loop
5. Grounded Answer

Use a thin divider between the offline index in Stage 2 and the online per-question pipeline in Stages 3-5. Keep arrows orthogonal or gently curved and prevent unrelated crossings.

MANDATORY TECHNICAL CORRECTIONS

Stage 1:
- Show historical frames above and exactly four blue-outlined current frames below.
- Use the exact labels "Causal boundary", "inaccessible future", "Historical frames", "Exact Recent-4", and "excluded from history index".
- The Exact Recent-4 blue arrow must never enter L0, L1, Search, or Working Evidence.

Stage 2:
- Show "L0 RawFrameMemory" with "frames · timestamps · CLIP embeddings".
- Show two parallel branches: "Temporal coverage" and "Multi-span CLIP proposals".
- Under proposals, use only the short labels "spans 1 / 2 / 4", "per-span MAD", and "temporal NMS".
- Add the concise note "CLIP locates change; it does not classify events".
- Merge both branches into "Block pack · ≤ 8 frames", then "Frozen VLM Perception".
- Split the VLM output into two genuinely separate paths:
  A. "Block overview" goes directly to searchable "OverviewMemory". It must bypass the Semantic Gate.
  B. "Proposal assessment" goes first to "Structural Validator" and then to "Semantic Gate".
- Draw one small repair loop from Structural Validator back to the VLM assessment labeled exactly "missing / duplicate / invalid → one repair". The repair is structural only and must not imply semantic correction or training.
- Semantic Gate must have exactly three outputs:
  "local event → searchable TransitionMemory"
  "global change → boundary metadata"
  "no meaningful change → reject"
- Only local events create searchable TransitionMemory. Global changes and rejected items must never appear in Search.
- Show L1 as "OverviewMemory + admitted TransitionMemory" with thin links back to L0 labeled "frame IDs".
- Use "L0 + L1 persistent memory". Do not show L2 or persistent Working Evidence.

Stage 3:
- Show "Question + options → EvidenceNeed Planner".
- Show only three compact example chips: "historical event", "attribute", and "before(A, B)".
- Add "preserve temporal semantics".
- Show Search over L1 only. State this with the exact label "Searchable L1: all overviews + admitted transitions".
- Draw BGE, BM25, and Conditional CLIP as three PARALLEL sibling branches that converge. They must not look sequential.
- Use exact branch labels:
  "BGE semantic"
  "BM25 lexical"
  "Conditional CLIP · attribute / spatial / state only"
- The convergence order must be exactly:
  "Absolute relevance gate · may return empty"
  → "Weighted RRF"
  → "Complementary selection"
  → "Working Evidence · temporary"
- Under Complementary selection, use only:
  "need coverage gain − redundancy"
- Search returns L1 memory IDs, not arbitrary raw frames.

Stage 4:
- Clearly separate the blue "LLM Tool Controller" from the gray "Validated Tool Executor".
- The Controller selects exactly one of: "Search L1", "Inspect L0", "Expand graph", "Compare nodes", "Finish".
- The Executor subtitle must be "causality · valid IDs · valid state · budgets".
- Draw one unmistakable closed loop:
  Controller → selected tool → Executor → Working Evidence update → reassess sufficiency → Controller.
- Inspect L0 must originate from a localized L1 memory and follow its frame IDs. Do not show arbitrary timestamp search.
- Expand follows graph edges; Compare uses two localized nodes.
- Show a small temporal-order inset only as:
  "locate A and B → verify endpoints → compare times → relation edge".
- Show all five budget badges:
  "≤ 4 tools"
  "≤ 2 searches"
  "≤ 2 expansions"
  "≤ 2 visual actions"
  "≤ 8 history frames"
- Add "Finish is free" and "No arbitrary timestamps or frame IDs".

Stage 5:
- Merge exactly two inputs: "Exact Recent-4 raw frames" and "Grounded historical evidence".
- Feed them to "Frozen VLM Reasoning", then "Answer".
- Keep the compact rule:
  "current state → Recent-4"
  "past events → grounded evidence"
- Rejected, unresolved, global-boundary metadata, and raw retrieval rankings must not enter final reasoning.

TYPOGRAPHY AND STYLE
- Correct the current misspellings "Autonemous" and "may retron empty". The intended phrases are "Autonomous" and "may return empty".
- Regenerate every label as real, correctly spelled English. No pseudo-words, malformed letters, accidental question marks, or clipped words.
- Use one modern sans-serif font, sentence case, horizontal text, consistent capitalization, and no text smaller than necessary for a two-column figure.
- Use short labels instead of prose. Keep all text within boxes with comfortable padding.
- Use flat vector-like shapes, thin 1.5-2 px strokes, consistent arrowheads, square or 4 px radius boxes, and generous white space.
- No gradients, shadows, 3D effects, decorative blobs, large icons, legends that repeat labels, or nested card styling.
- Reduce thumbnail detail before reducing font size.

COLOR SEMANTICS
- Exact Recent-4 and final reasoning: blue #356AE6.
- Coverage and OverviewMemory: teal #2A9D8F.
- Proposals and TransitionMemory: coral #E76F51.
- Working Evidence: pale lavender #EEE8F7 with outline #7A6F9B.
- Validation and executor: cool gray #E9EEF2.
- Rejection and inaccessible future: restrained red #C84B4B.
- Text: charcoal #263238; background: pure white.

DO NOT ADD
Training, fine-tuning, gradients, future-frame access, arbitrary timestamps, direct Search over L0, persistent L2/L3 memory, answer-written memory, cross-video memory, external web tools, fixed evidence-card limits, or any module not listed above.

OUTPUT
Return one polished wide scientific figure only. Do not add a decorative title, caption, watermark, conference logo, figure number, or AI-generation disclosure inside the image; the paper caption provides those separately.
```

### Typography-Only Follow-up

若重绘后的结构正确但仍有少量文字问题，把新图再次作为 edit target，并使用：

```text
Perform a typography-only correction pass on the attached VeriStream method figure. Preserve every box, arrow, position, color, spacing relationship, and technical connection exactly. Correct all misspelled, malformed, clipped, or pseudo-English text using the exact labels from the previous prompt. In particular verify "Autonomous", "Absolute relevance gate · may return empty", "Structural Validator", "OverviewMemory", "TransitionMemory", "Working Evidence", "Exact Recent-4", and all five budget badges. Do not redraw, add, remove, merge, or reconnect any component. Return the corrected wide figure only, with no caption, watermark, figure number, or decorative title.
```
