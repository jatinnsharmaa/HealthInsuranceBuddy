# Chat UX Improvements — Design Spec
**Date:** 2026-05-28  
**Scope:** Frontend only (`components/MessageBubble.tsx`, `components/ChatPane.tsx`, `next.config.*`)

---

## Problem Summary

Four issues observed in the live chat UI:

1. **Reasoning step labels merge** — "Step 2: RETRIEVEStep 3: EVALUATE_POLICY" rendered as one string; internal agent chain-of-thought leaks into the answer body.
2. **No follow-up question nudges** — after a response, the user sees no suggestions for what to ask next; chat feels terminal.
3. **Next.js dev indicator (N circle)** — bottom-left development badge visible in the app.
4. **PDF page accuracy** — the LLM occasionally cites the wrong page number. This is a retrieval/prompt concern, *not* addressed in this spec.

---

## Design

### 1. Reasoning / Answer Split in `MessageBubble`

**Goal:** Cleanly separate the agent's internal reasoning trace from the answer the user cares about.

#### Content parsing

The orchestrator prompt always produces output in this structure:

```
Step 1: THINK
<reasoning>

Step 2: RETRIEVE
Step 3: EVALUATE_POLICY
<evaluation>

Step 5: GENERATE

Verdict: Yes / No / Partial

<answer prose>

Sources:
- ...

Disclaimer: ...
```

Split rule: everything *before* `\nVerdict:` (case-insensitive) is the **reasoning block**; everything from `Verdict:` onwards is the **answer block**.

If no `Verdict:` is present (partial stream, or unusual LLM output), treat the entire content as the answer block.

#### During streaming

Show a pulsing `<ThinkingIndicator>` component above the answer area:

```
⚙  Evaluating policy…  ▌
```

The label cycles through step names as they appear in the stream:
- "THINK" → "Thinking…"
- "RETRIEVE" → "Retrieving clauses…"
- "EVALUATE_POLICY" → "Evaluating policy…"
- "GENERATE" → "Generating answer…"
- default → "Working…"

Detect the current step by scanning the latest streamed content for the most recent `Step N: STEPNAME` match.

The answer block (everything post-`Verdict:`) streams in normally below the indicator.

#### After streaming completes

- `<ThinkingIndicator>` is replaced by a `<ThinkingDisclosure>` toggle.
- Default state: **collapsed** — renders as a single muted line:
  ```
  💭 View reasoning  ›
  ```
- Expanded state: reasoning text rendered in a `<pre>`-like block with `font-mono text-xs text-slate-400 bg-slate-50 rounded p-3 whitespace-pre-wrap`. Step labels (`Step N: STEPNAME`) are bolded.
- The answer block is rendered at full prominence using the existing `renderContent()` path.

#### Newline fix

`renderContent()` currently returns React elements from `.split()` — `\n` characters in plain text spans are lost because they're inside `<span>` without `whitespace-pre-wrap`. Fix: wrap each plain-text span output in a fragment that converts `\n` to `<br />`, or apply `whitespace-pre-wrap` to the answer container `<div>`.

---

### 2. Follow-Up Question Suggestions

**Goal:** After each completed agent response, surface 2–3 contextual question pills to lower the "what do I ask next?" friction.

#### Placement

Rendered inside `MessageBubble`, below the feedback buttons, only when `!message.isStreaming && isLast`.

#### Topic detection

A pure function `detectTopic(question: string, answer: string): string` maps keyword presence to a topic slug. Checked in order (first match wins):

| Topic slug | Keywords (question OR answer) |
|---|---|
| `waiting_period` | waiting period, ped, pre-existing, 36 month |
| `maternity` | maternity, pregnancy, newborn, delivery |
| `exclusion` | exclusion, not covered, excluded, permanent |
| `hospital_network` | cashless, hospital, network, tpa |
| `claim` | claim, reimbursement, discharge, document |
| `room_rent` | room rent, sub-limit, icu, ward |
| `premium` | premium, renewal, increase, portability |
| `(fallback)` | — |

#### Suggestion map

```ts
const FOLLOW_UP_MAP: Record<string, string[]> = {
  waiting_period: [
    "Does portability reduce my waiting period?",
    "What counts as a pre-existing disease?",
    "Can the waiting period be waived?",
  ],
  maternity: [
    "Is there a waiting period for maternity cover?",
    "Are newborn expenses covered from day one?",
    "What's the sub-limit for maternity claims?",
  ],
  exclusion: [
    "Are there permanent exclusions that can never be covered?",
    "What's the room rent sub-limit?",
    "Are dental and vision covered?",
  ],
  hospital_network: [
    "How do I find a network hospital near me?",
    "What happens if I go to a non-network hospital?",
    "Is pre-authorisation always required?",
  ],
  claim: [
    "What documents are needed for a reimbursement claim?",
    "What's the deadline for filing a claim?",
    "Does the insurer have a cashless claim process?",
  ],
  room_rent: [
    "How does the room rent sub-limit affect my claim?",
    "Is ICU treatment covered without a sub-limit?",
    "Can I upgrade my room if I pay the difference?",
  ],
  premium: [
    "Will my premium increase after a claim?",
    "How does portability work?",
    "Is there a no-claim bonus?",
  ],
  fallback: [
    "What are the main exclusions in this policy?",
    "Is there a room rent sub-limit?",
    "Can I add a family member mid-term?",
  ],
};
```

Show the first 2 suggestions from the matched topic (3 if fallback). Clicking a pill calls `onSend(question)` — same handler as the input bar.

#### Styling

Pills are small, rounded, outlined buttons with a `→` prefix:

```
→ Does portability reduce my waiting period?    → What counts as a pre-existing disease?
```

Class: `text-xs border border-slate-200 rounded-full px-3 py-1 text-slate-600 hover:border-blue-300 hover:text-blue-700 hover:bg-blue-50 transition-colors`

---

### 3. Hide Next.js Dev Indicator

In `next.config.ts` (or `.js`), add:

```ts
const nextConfig = {
  // existing config...
  devIndicators: false,
};
```

---

## Files Changed

| File | Change |
|---|---|
| `components/MessageBubble.tsx` | Add `ThinkingIndicator`, `ThinkingDisclosure`, `FollowUpSuggestions`; split content into reasoning/answer zones; fix `\n` in `renderContent` |
| `components/ChatPane.tsx` | Pass `onSend` down to `MessageBubble` for follow-up pill clicks |
| `next.config.ts` / `next.config.js` | Add `devIndicators: false` |

No backend changes. No new dependencies.

---

## Out of Scope

- PDF page number accuracy (LLM/retrieval concern, separate issue)
- AI-generated follow-up questions (deferred — requires backend SSE change)
- Mobile/responsive layout

---

## Success Criteria

- [ ] Reasoning steps never appear as raw body text
- [ ] During streaming, a "Thinking…" indicator shows the current step
- [ ] After streaming, reasoning is collapsed behind a toggle
- [ ] 2–3 contextual follow-up pills appear below every completed response
- [ ] Next.js "N" badge is gone
- [ ] All existing functionality (citation links, verdict badge, feedback buttons, live data tag) still works
