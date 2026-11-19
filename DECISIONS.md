# Approach

**Design.** One canonical record, one config per company. Every company's raw files are mapped by a YAML config
into the same record (date, supplier, vendor id, description, item code, amount, currency, status, PO, invoice,
GL text...). Everything after ingest is company-agnostic, and no category name appears in code: the tree is read
from a file with up to four levels. All decisions are written to columns (`exclusion_rule`, `match_kind`,
`class_method`), so any row can be explained. Reruns are byte-identical.

**Stages.**
1. *Clean.* Amounts (thousand separators, decimal commas, negatives), VAT (`15`, `0.15`, `15%`, blank -> 15%, explicit `0`
   stays 0; net = gross / (1+rate) only for gross sources), mixed date formats (ambiguous day/month resolved per row),
   FX (line rate if present, else the FX table for the invoice month, nearest month for gaps).
2. *Exclusions.* Status mapping, then rules, then duplicates; first match wins. Company rules (B: intercompany GL accounts,
   supplier types, billing references) come from comparing the seed labels with the data; generic conservative rules
   (tax labels, intercompany, royalties, interest) apply everywhere.
3. *Matching (A).* Invoice-tool lines are matched to ERP receipts by PO, amount (net of VAT) and OCR-tolerant text/supplier
   similarity; a missing PO is recovered from another line of the same supplier + invoice number, or from the PO range of
   neighbouring record ids. Matched lines are `DUPLICATE` and inherit the ERP record's keys; repeated exports collapse.
4. *Keys.* Suppliers: normalised names merged by similarity, plus vendor ids (capped for collective accounts), with an
   optional lookup of labelled keys. Items: the item code, else normalised text linked to a coded item.
5. *Classification.* Supervised linear model on text + supplier + context with calibrated confidence when labels exist;
   with no labels, an LLM (Groq) labels unique items against the tree (level by level for big trees), and a
   supplier-context model trained on those labels fills gaps and gives a second opinion; offline fallback: tree words +
   self-training.

**Measured quality** (all from the provided data; A labels hidden or held out, grouped by text so repeats cannot leak):

| What | Result |
|---|---|
| B exclusions vs 3,000 seed labels | 3,000 / 3,000 agree (450 excluded, 2,550 included) |
| A: tool lines found by PO + amount | 16,743 of 17,620 with a PO in the ERP (95.0%); 22,010 tool lines marked `DUPLICATE` in total |
| A: matching without a PO (PO hidden on 18% of known matches) | 95.6% re-found, right ERP PO 90.6%; 7.4% of tool-only-looking lines wrongly matched (upper bound) |
| A suppliers, lines with a resolved supplier (16,138) | pairwise precision 1.000, recall 1.000 (names alone) |
| A items with a code, excluding keys spanning several codes | precision 1.000, recall 1.000; text only (codes hidden): 0.970 / 1.000 |
| A classification, supervised, 20% held out | L1 99.1%, L2 96.2%, L3 93.0% (spend-weighted L3 81.6%) |
| B classification, supervised from seeds, 25% held out | L1 88.8%, L2 83.6%, L3 76.4% (spend-weighted L3 87.2%) |
| A classification, **no labels**, local only | L1 50.6%, L2 17.9%, L3 16.8% |
| B classification, **no labels**, local only | L1 48.3%, L2 46.4%, L3 28.8% |
| A classification, **no labels**, with Groq (`openai/gpt-oss-120b`, 300-item limit) | `llm`: L1 67.9%, L3 40.4%, spend-weighted L3 59.4% on 18,302 labelled lines; combined output: L1 52.7%, L3 24.3%, spend-weighted L3 51.0%; first-pass coverage 100% (259/259), plus 41 cached answers |

Zero-label quality is the weak point: category names rarely resemble abbreviated descriptions
(`BLD,SEAL;UG SPRT WRK`), which is why an LLM layer exists for the unseen company. Local sentence embeddings were
tested and made zero-label results worse on both companies (A L3 5.8%, B L3 20.9%), so they are off.
With a simulated 75%-accurate LLM, the supplier-context second opinion lifted A's L3 from 62.8% to 71.4%;
that is a simulation (random errors), not a result, and the live figure should replace it.

**Runtime.** A (65,549 records): about 80-110 s on a laptop, 60+ of them classification. B (41,028): under 15 s.
The brief's limit is 60 minutes for 25,000 lines. With the LLM the time is set by the provider's rate limit.
The live-run output did not capture elapsed wall time, so seconds per 1,000 items is not reported.

**LLM cost.** Cost per 1,000 lines = (unique items per 1,000 lines) x (tokens per item) x (price per token).
Only unique items are sent, largest spend first, in batches of 40, with answers cached. In the live run, 259
uncached item answers used 22,203 tokens (85.7 tokens per newly requested item); 41 further answers came from the
cache. The 300-item limit equals 4.58 items per 1,000 input records. No provider price was captured, so a monetary
cost per 1,000 lines is not reported.
Without a key the tool runs the local method; the run is still valid, only less accurate.

**Limits.** There are no ground-truth labels for tool-to-ERP matches, so matching is validated by hiding known POs.
The supplier lookup assumes the hidden truth is built like the provided labels (see DECISIONS.md D-20). The PO-range
trick depends on record ids rising with PO number. A's largest classification losses are contract-type leaves
(e.g. "CM Rand Per Tonne Agreement") that the description does not show.
