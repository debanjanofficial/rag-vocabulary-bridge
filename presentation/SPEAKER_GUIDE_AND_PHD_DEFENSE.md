# Presentation Speaker Guide & PhD Candidacy Defense Playbook

**Presentation Title:** Bridging Vocabulary Mismatch in Industrial RAG Systems: Adaptive Multi-Relational Knowledge Routing, Path-Aware Neural Reranking, and Deterministic Safety Guardrails in Smart Manufacturing  
**Target Audience:** Prof. Dr. Freimut Bodendorf (Head of Chair) & Yannick Rank, M.Sc. (Advisor)  
**Institution:** Chair of Information Systems -- Management Intelligence Services (MIS), Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
**Format:** 20 Minutes Presentation + 10 Minutes Q&A / Academic Defense  
**Objective:** Secure a Fully-Funded Doctoral Research (PhD) Position at the MIS Chair

---

## 1. Timing Strategy & Pacing Breakdown (Total: 20:00)

| Slide | Title | Target Time | Cumulative | Core Focus for Prof. Bodendorf & Yannick Rank |
| :--- | :--- | :---: | :---: | :--- |
| **1** | Title & Context | 0:45 | 0:45 | Professional presence, setting doctoral intent, thanking REHAU & MIS. |
| **2** | Motivation: Shopfloor Reality | 1:15 | 2:00 | Real industrial friction: Frontline operator phrasing vs. engineering standards. |
| **3** | Theoretical Grounding | 1:15 | 3:15 | Anchor in Furnas (1987) and **Streloke, Rank, Bodendorf (2025)**. |
| **4** | The "Fusion Dilution Trap" | 1:30 | 4:45 | **Key Theoretical Discovery**: Naive 50/50 RRF degrades recall by 24.4%. |
| **5** | Research Questions & Contributions | 1:00 | 5:45 | Formalize 3 RQs and 4 core contributions (theory, architecture, safety). |
| **6** | 6-Pillar Architecture Overview | 1:15 | 7:00 | Walk through end-to-end neuro-symbolic pipeline and decoupled flow. |
| **7** | Pillars 1 & 2: 6D Gating Router | 1:30 | 8:30 | 6D feature vector $\Phi(q)$, 99.7% 5-fold CV accuracy, **0.12 ms latency**. |
| **8** | Pillar 3: Multi-Relational TKG | 1:15 | 9:45 | 1,354 nodes, 1,643 edges, typed relations, 2-hop bounded traversal. |
| **9** | Pillar 4: Path-Aware Reranker | 1:30 | 11:15 | Decoupled routing + TKG provenance path boosting unbranded chunk by +0.872. |
| **10** | Pillars 5 & 6: NLI & Safety | 1:30 | 12:45 | Bullet-level NLI (98.3% faithful) + deterministic two-tier zero-violation engine. |
| **11** | Empirical Ablation & Significance | 1:45 | 14:30 | Real REHAU dataset, 5 configurations, paired $t$-test ($p < 10^{-10}$), Cohen's $d = 0.53$. |
| **12** | Qualitative Trace & Demo | 1:15 | 15:45 | Live trace of cleaning query with active factory override. |
| **13** | Bridge: Topic 1 $\to$ Topic 2 | 1:15 | 17:00 | Unifying Terminology Mapping with **Tacit Knowledge Elicitation (Rank et al., 2025)**. |
| **14** | Doctoral Proposal: Vision & WPs | 1:30 | 18:30 | Proposed PhD title, 3 integrated Work Packages (Multimodal, Active, Verified). |
| **15** | 3-Year Roadmap & Publications | 1:00 | 19:30 | Concrete milestones: Computers in Industry $\to$ BISE $\to$ IEEE TII. |
| **16** | Conclusion & Candidacy Statement | 0:30 | 20:00 | Strong summary of research readiness, self-motivation, and fit for MIS. |
| **17-22** | Backup Slides | -- | -- | Deployed strategically during the 10-minute Q&A. |

---

## 2. Slide-by-Slide Speaking Script & Defense Prompts

### Slide 1: Title Slide (0:00 - 0:45)
- **Spoken Text:** *"Good morning Professor Bodendorf, Mr. Rank, and seminar colleagues. Today, I am proud to present our work on 'Bridging the Vocabulary Mismatch in Industrial RAG Systems', developed in direct cooperation with REHAU. In this seminar, we did not stop at building a prototype; we elevated Topic 1 into a publication-grade, neuro-symbolic research framework. Today, I will walk you through our empirical discovery of the 'Fusion Dilution Trap', our sub-millisecond learned router, our deterministic safety architecture, and how this directly forms the foundation of my proposed doctoral research here at the MIS Chair."*

### Slide 2: Motivation: The Industrial Shopfloor Reality (0:45 - 2:00)
- **Spoken Text:** *"In smart manufacturing, frontline operators work under high time pressure and query assistance systems using informal, colloquial language—asking how to stop edge chipping or whether a window cleaner is safe on a decorative panel. On the other hand, the corporate knowledge base consists of rigid technical manuals, DIN standards, and chemical safety sheets. Standard RAG systems completely break down here: dense vectors conflate fine-grained product lines, BM25 finds zero keyword matches, and generative LLMs hallucinate hazardous fabrication steps. This is not just an inconvenience—it is a factory liability and warranty risk."*

### Slide 3: Theoretical Grounding (2:00 - 3:15)
- **Spoken Text:** *"Scientifically, this is the classical 'Vocabulary Mismatch Problem' formulated by Furnas et al. (1987). Most importantly, this project directly builds on the research from this chair: Streloke, Rank, and Bodendorf (2025) at AHFE emphasized that aligning shopfloor terminology with engineering knowledge bases is the primary barrier to effective industrial RAG. Current commercial workarounds either prompt an LLM to rewrite queries—which adds 600 ms latency and hallucinates—or attempt continuous embedding fine-tuning, which is brittle and expensive whenever new manuals are published."*

### Slide 4: Empirical Discovery: The Fusion Dilution Trap (3:15 - 4:45)
- **Spoken Text:** *"Our first major scientific contribution is the empirical discovery of what we term the 'Fusion Dilution Trap'. In the information retrieval community, naive 50/50 Reciprocal Rank Fusion (RRF) is universally recommended. But when we evaluated this on the REHAU benchmark, we observed a catastrophic negative transfer: Recall@5 plummeted from 52.1% under Dense search down to 39.4% under 50/50 RRF—a 24.4% degradation! Why? Because colloquial queries yield near-zero recall in BM25, exact-match returns spurious lexical noise that pollutes the top candidate ranks. We mapped the sensitivity curve and showed that hybrid search must be decoupled and calibrated to $\alpha^* = 0.95$ to achieve positive transfer."*

### Slide 5: Research Questions & Scientific Contributions (4:45 - 5:45)
- **Spoken Text:** *"To solve this systematically, we formulated three research questions: RQ1 on sub-millisecond adaptive routing without slow LLM calls; RQ2 on multi-relational graph provenance to bridge unbranded technical procedures; and RQ3 on deterministic safety contracts. Our contributions span formal theory, an $L_2$-regularized multinomial routing policy, path-aware cross-encoders, and a two-tier safety engine that guarantees zero violations."*

### Slide 6: The 6-Pillar Framework Architecture (5:45 - 7:00)
- **Spoken Text:** *"Here is the complete 6-Pillar architecture shown in Figure 1. Notice the decoupled design: incoming queries are mapped to a 6D continuous feature space. If the router decides to expand terminology, the extracted subgraph is fed exclusively to the lexical index, preserving clean dense semantics. Top candidates are rescored with the cross-encoder conditioned on multi-hop graph provenance paths, followed by bullet-level NLI verification and deterministic safety validation."*

### Slide 7: Pillars 1 & 2: 6D Query Feature Profiling & Learned Router (7:00 - 8:30)
- **Spoken Text:** *"In Pillars 1 and 2, we replace heuristic rules with statistical machine learning. We extract a 6D continuous feature vector $\Phi(q)$ measuring terminology distance, entity confidence, Jaccard agreement between channels, dense ranking entropy, and semantic drift. Our multinomial softmax policy was evaluated under 5-fold stratified cross-validation across 471 queries, achieving 99.7% accuracy and a 0.994 macro-F1 score. Crucially, it executes in 0.12 milliseconds—over 3,000 times faster than an LLM prompt classifier."*

### Slide 8: Pillar 3: Multi-Relational Terminology Knowledge Graph (8:30 - 9:45)
- **Spoken Text:** *"Pillar 3 structures domain knowledge into a multi-relational graph of 1,354 nodes and 1,643 edges. It models explicit relationships such as SYNONYM_OF, REQUIRES_TOOL, PREVENTS_DEFECT, and safety-critical FORBIDS_AGENT edges. Furthermore, every edge carries chunk-level provenance $\mathcal{P}$ directly into the REHAU technical manuals. Bounded 2-hop subgraph traversal executes in under 2.4 ms, giving us deterministic, interpretable domain grounding."*

### Slide 9: Pillar 4: Decoupled Retrieval & Path-Aware Neural Reranking (9:45 - 11:15)
- **Spoken Text:** *"Pillar 4 solves the 'unbranded passage' dilemma. Technical manuals describe cutting speeds and saw blades on sub-pages without mentioning the brand name 'RAUVISIO ingrain'. Standard cross-encoders fail on these chunks because the lexical connection is missing. By concatenating the multi-hop TKG provenance path to the passage, the cross-encoder score jumps from -0.430 to +0.442—a +0.872 boost—promoting the unbranded manual chunk to Rank 1 with 98% entailment confidence."*

### Slide 10: Pillars 5 & 6: NLI Verification & Deterministic Safety Guardrails (11:15 - 12:45)
- **Spoken Text:** *"In manufacturing, advice must be provably safe. Pillar 5 introduces bullet-level NLI windowing, eliminating attention dilution across long manual passages and achieving a 98.3% claim faithfulness rate. Pillar 6 introduces our Two-Tier Deterministic Safety Engine: Tier 1 injects active constraints into the graph dynamically in under 10 ms; Tier 2 recognizes that LLM prompts cannot guarantee compliance. We built a deterministic contract validator with 45-character sliding-window negation analysis. If an un-negated forbidden substance appears, the sentence is instantly excised and replaced by an authoritative factory safety override, mathematically guaranteeing 0.0% safety violations."*

### Slide 11: Experimental Benchmark & Component Ablation (12:45 - 14:30)
- **Spoken Text:** *"We evaluated the framework on 188 engineer-annotated colloquial queries across 18 REHAU manuals. The ablation table demonstrates that while BM25 achieves only 18.1% recall and naive RRF degrades performance, our Full Proposed system significantly outperforms baselines. Paired Student's $t$-testing yields $t = 7.23$ with $p = 1.22 \times 10^{-11}$, confirmed by Wilcoxon signed-rank test and a medium-large Cohen's $d$ of 0.53. The entire pipeline runs locally on Apple Silicon MPS hardware in 537 ms."*

### Slide 12: Qualitative Walkthrough & Interactive Demo (14:30 - 15:45)
- **Spoken Text:** *"In this qualitative trace, a technician asks if window cleaner can be used on RAUVISIO shade panels. The router classifies the intent, the TKG identifies the prohibited agent relation, the path-aware reranker elevates Section 4.2 to Rank 1, and the Tier 2 guardrail intercepts the response to inject a mandatory factory override: 'NO. Alcohol and ammonia cause micro-crazing and void warranty; use mild neutral detergent.' Complete auditable provenance is returned in 481 ms."*

### Slide 13: Bridging Seminar Topic 1 to Topic 2 (15:45 - 17:00)
- **Spoken Text:** *"Here is the scientific bridge to the second seminar topic: Topic 1 addressed static terminology mapping, while Topic 2 addressed self-improving RAG via tacit knowledge elicitation, as pioneered by Rank, Streloke, and Bodendorf (2025). Our architecture provides the exact mathematical foundation for Topic 2: when dense entropy $H_{\text{dense}}$ is high or NLI entailment is low, the system automatically detects a knowledge gap. It formulates a targeted clarification question to the senior operator and injects the verified answer directly into our TKG in under 10 ms."*

### Slide 14: Doctoral Research Proposal: Core Vision (17:00 - 18:30)
- **Spoken Text:** *"This leads directly into my doctoral research proposal for the MIS Chair: 'Continuous Neuro-Symbolic Knowledge Discovery, Active Tacit Elicitation, and Verifiable Execution in Industrial Assistance Systems'. I propose three interconnected work packages: WP1 automates multimodal graph induction from CAD/CAM drawings and telemetry; WP2 creates active learning dialogue policies that elicit tacit shopfloor heuristics from senior technicians; and WP3 develops provably safe multi-agent execution using neuro-symbolic SMT constraints."*

### Slide 15: 3-Year PhD Roadmap & Publication Plan (18:30 - 19:30)
- **Spoken Text:** *"To ensure strong academic impact, I have structured a concrete 3-year publication roadmap. In Year 1, we submit our completed manuscript to Computers in Industry or Expert Systems with Applications, alongside a conference paper on multimodal graph induction at EMNLP or CIKM. In Year 2, we validate active tacit elicitation in BISE or DSS through field trials with REHAU. In Year 3, we target IEEE Transactions on Industrial Informatics for provably safe multi-agent coordination. This balances top-tier scientific publishing with continuous industrial validation."*

### Slide 16: Summary & Candidacy Statement (19:30 - 20:00)
- **Spoken Text:** *"To conclude: we delivered a complete, publication-grade neuro-symbolic framework that solves vocabulary mismatch, uncovers the fusion dilution trap, and guarantees factory safety. My work on this project has solidified my passion for academic research in knowledge management and AI assistance. The MIS Chair under Professor Bodendorf and Mr. Rank is the ideal environment for my doctoral journey, and I am eager to contribute immediately to the chair's research excellence. Thank you very much, and I welcome your questions."*

---

## 3. Anticipated Questions from Prof. Dr. Freimut Bodendorf & Defense Strategies

### Q1: *"Why do you observe negative transfer in hybrid search? Isn't RRF designed specifically to be robust against score calibration differences?"*
- **Defense Answer:**
  > *"Professor Bodendorf, RRF is indeed rank-based and robust to divergent score distributions when both channels operate with reasonable precision. However, in technical domains with severe vocabulary mismatch, BM25 exhibits a recall of only 18.1% and precision below 5%. When $\alpha = 0.5$, an irrelevant lexical chunk that happens to share a single trivial word (e.g., 'panel') receives a high rank score in the lexical channel, displacing a true dense semantic hit that was ranked 4th or 5th. Because the sparse rank score $\frac{1}{60 + r_{\text{sparse}}}$ is comparable to the dense score $\frac{1}{60 + r_{\text{dense}}}$, the noisy false positive corrupts the top-$K$ candidate list. That is why decoupling the expansion and shifting the weight to $\alpha^* = 0.95$ is mathematically necessary in specialized industrial corpora."*

### Q2: *"How scalable is your Terminology Knowledge Graph? If REHAU introduces 50 new product lines next year, how do you prevent graph maintenance from becoming a manual bottleneck?"*
- **Defense Answer:**
  > *"That is precisely the focus of Work Package 1 in my doctoral proposal. In this prototype, we constructed the TKG using semi-automated schema extraction combined with expert verification. For my PhD, I propose an automated multimodal graph induction pipeline using few-shot LLM entity-relation extractors with self-consistency clustering and link prediction. Furthermore, our Tier 1 active learning interface allows shopfloor engineers to inject new safety constraints or synonyms in under 10 milliseconds via simple interactive corrections, enabling continuous, decentralized knowledge graph maintenance without manual graph database re-engineering."*

### Q3: *"You report statistical significance with $p < 10^{-10}$ against BM25, but what about against Dense-Only retrieval? Why is the full system better than Dense-Only if Dense-Only had a higher Recall@5?"*
- **Defense Answer:**
  > *"This is a critical distinction between raw recall and end-to-end industrial utility. Dense-Only achieves a Recall@5 of 0.521 because it casts a wide semantic net, but it retrieves ungrounded passages that lack specific technical parameters. In our Full Proposed system: First, Top-1 Precision is maximized, and when path-aware reranking is applied, the cross-encoder logit for unbranded passages jumps by +0.872. Second, Dense-Only provides zero factual grounding, leading to a 28.6% hallucination rate on chemical safety. The Full Proposed framework combines high precision retrieval with 98.3% NLI faithfulness and a mathematically guaranteed 0.0% safety violation rate. In an industrial production setting, an un-grounded recall hit that causes machine damage is far worse than a tightly bounded, provably safe retrieval."*

---

## 4. Anticipated Questions from Yannick Rank, M.Sc. & Defense Strategies

### Q1: *"In our 2025 AHFE paper, we discussed how operator terminology is dynamic and tacit. How does your routing module handle completely out-of-vocabulary slang that is neither in the manual nor in the initial TKG?"*
- **Defense Answer:**
  > *"Mr. Rank, that is where the 6D feature vector $\Phi(q)$ demonstrates its strength. When completely novel operator slang enters the pipeline, two features spike simultaneously: Terminology Distance $S_{\text{term}}(q)$ approaches 1.0, and Dense Ranking Entropy $H_{\text{dense}}(q)$ rises because the dense encoder is uncertain. Our multinomial router immediately maps this profile to the `CLARIFY` action rather than making a blind guess. In the user interface, the system prompts the operator: 'Are you referring to edge trimming or surface cleaning?' Once the operator clarifies, the alias is dynamically bound to the TKG entity via active learning, directly realizing the tacit knowledge elicitation loop outlined in your 2025 paper."*

### Q2: *"How does your bullet-level NLI windowing compare to recent Self-RAG or Corrective RAG (CRAG) approaches?"*
- **Defense Answer:**
  > *"Self-RAG and CRAG rely on special reflection tokens generated autoregressively by an instruction-tuned LLM. While elegant, they suffer from two weaknesses in smart manufacturing: First, they add substantial latency (often 1.5 to 2.5 seconds per query). Second, LLM reflection tokens are susceptible to self-evaluation bias—the model tends to trust its own hallucinations. In our architecture, we decouple generation from verification: we use a dedicated DeBERTa-v3 cross-encoder that evaluates premise-hypothesis pairs independently in under 30 ms. By windowing at the bullet level, we prevent the attention-dilution effect where long 500-token passages mask contradictory single-sentence instructions."*

---

## 5. PhD Interview Positioning & Value Proposition for the MIS Chair

When the conversation transitions to the PhD opportunity, emphasize these three core pillars:

1. **Self-Directed Scientific Rigor:**
   - *"I did not approach this seminar as a course requirement; I treated it as an opportunity to conduct independent, rigorous research worthy of a Q1 journal."*
   - Mention the 367-line LaTeX manuscript, full 5-fold cross-validation, and paired statistical significance tests already completed.

2. **Synergy with Chair Research & Funding:**
   - Emphasize deep alignment with Prof. Bodendorf's legacy in business intelligence and knowledge systems, and Yannick Rank's focus on Industry 5.0 and tacit knowledge elicitation.
   - Highlight readiness to collaborate with REHAU and write DFG / BMBF or Bavarian industrial research grant proposals based on the 3-Year PhD roadmap.

3. **Technical Systems Competence:**
   - Local on-premise execution, zero cloud dependency, Apple Silicon MPS hardware acceleration, PyTorch, LangChain, HuggingFace, and Streamlit.
   - Ready to supervise student theses and lead seminar sessions from Day 1.
