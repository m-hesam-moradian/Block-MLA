# **Strategic Dataset Selection and Experimental Architecture for Evaluating Block-MLA in Decentralized AI Ecosystems** 

## **1. Introduction to the Asynchronous Distributed Low-Communication Paradigm** 

The training of Large Language Models (LLMs) has historically relied on centralized, tightly coupled synchronous infrastructure, such as homogenous GPU clusters connected via high-bandwidth interconnects like NVLink or InfiniBand<sup>1</sup> . However, the astronomical costs and physical limitations of co-locating thousands of accelerators have catalyzed the development of Distributed Low-Communication (DiLoCo) optimization frameworks. The DiLoCo paradigm reconceptualizes data parallelism as a bilevel optimization problem. In this framework, localized worker nodes execute hundreds of inner-loop steps using optimizers like AdamW before communicating pseudo-gradients to a global server, which then applies an outer-loop optimizer, typically utilizing Nesterov momentum<sup>1</sup> . By communicating only the model weights and not the intermediate gradients, synchronous DiLoCo drastically reduces communication overhead, often by a factor of 500, allowing for cross-datacenter and even intercontinental training pipelines<sup>1</sup> . 

While synchronous DiLoCo mitigates bandwidth constraints, it remains fundamentally vulnerable to the straggler effect, where the entire cluster is bottlenecked by the computational pace of the slowest node. Asynchronous variants of DiLoCo address this by eliminating synchronization barriers, allowing workers to continuously compute and communicate at their intrinsic hardware speeds<sup>3</sup> . Yet, this architectural freedom introduces a severe optimization pathology known as gradient staleness. Workers operating asynchronously compute updates based on outdated global model states, generating stale pseudo-gradients that can conflict directionally with the current trajectory of the global model<sup>3</sup> . This directional conflict is severely exacerbated under conditions of system heterogeneity and 

non-Independent and Identically Distributed (non-IID) data routing across the worker nodes<sup>10</sup> . Standard Momentum Look-Ahead (MLA) algorithms attempt to correct this staleness by extrapolating the negative momentum direction uniformly across the entire pseudo-gradient<sup>3</sup> . However, the proposed draft for "Block-MLA: Dynamic Tensor-Aware Momentum Look-Ahead" correctly identifies that uniform correction is fundamentally insufficient under non-IID conditions. Different parameter blocks within a neural network—such as attention projection matrices versus multilayer perceptron (MLP) weights or embedding tables—exhibit varying degrees of alignment with the global trajectory<sup>3</sup> . The Block-MLA algorithm addresses this geometric distortion by dynamically calculating extrapolation coefficients at the tensor-block 

level based on directional compatibility scores, aggressively dampening conflicting components while preserving aligned optimization signals without requiring explicit orthogonal projections<sup>3</sup> . 

To empirically validate the Block-MLA algorithm, a meticulously designed experimental framework is required. The simulation must forcefully induce the precise conditions that Block-MLA is engineered to solve: extreme gradient staleness combined with severe non-IID data distribution, which ultimately leads to representation drift<sup>3</sup> . This report provides an exhaustive analysis of the optimal datasets, mathematical partitioning strategies, and implementation architectures necessary to construct a robust, single-GPU simulated environment that will conclusively demonstrate the efficacy of Block-MLA against established baselines. 

## **2. Theoretical Framework of the Block-MLA Simulated Experiment** 

The experimental design specified for evaluating Block-MLA relies on a single-GPU simulated environment to bypass the extreme engineering complexities of deploying a physical multi-node wide-area network (WAN) cluster<sup>3</sup> . By virtualizing the cluster, researchers can strictly control the isolated variables of staleness, hardware heterogeneity, and data routing. 

The simulation architecture defines heterogeneous workers<sup>3</sup> . To model severe straggler effects typical of decentralized physical infrastructure networks, the computational pace configurations, representing seconds per inner step, are drawn from a disparate set of values<sup>3</sup> . The critical experimental configuration highlighted in the draft is the extreme staleness profile, where the paces are defined as 3. In this specific profile, Worker 1 operates at a pace of 1 second per step, while Workers 2 through 5 require 15 seconds per step. Consequently, Worker 1 will submit 15 fresh pseudo-gradients to the global outer optimizer in the exact time it takes the other four workers to submit a single, highly stale update. 

This extreme temporal imbalance creates a fragile optimization environment. If the dataset partitioned among these workers is purely IID, a stale gradient from a slow worker will generally point in the same expected geometric direction as the fresh gradients from the fast worker, differing primarily in magnitude and local noise. In a purely IID scenario, standard global momentum correction or clipping mechanisms are often sufficient to maintain convergence<sup>1</sup> . However, if the data is distributed in a non-IID manner, the global model will rapidly overfit to the specific linguistic or domain characteristics of Worker 1. When the updates from Workers 2 through 5 finally arrive, they will be heavily delayed and geometrically misaligned with the fast worker's loss basin, creating a directional conflict that can shatter the global momentum<sup>3</sup> . The evaluation specifies a decoder-only Transformer model, styled after the NanoGPT architecture, parameterized between 15 million and 90 million parameters<sup>3</sup> . Operating at this scale allows the single-GPU simulator to maintain all worker states, including local model replicas, inner AdamW optimizer states, and communication buffers, in the GPU VRAM 

simultaneously, preventing crippling CPU-to-GPU memory transfer bottlenecks<sup>19</sup> . This scale is also sufficiently deep to exhibit the tensor-block divergence that Block-MLA capitalizes on. The 

training regimen dictates an inner learning rate of using the AdamW optimizer, and 

an outer learning rate of , simulating 300 outer steps consisting of 80 inner steps each<sup>3</sup> . For Block-MLA to demonstrate a definitive algorithmic advantage over standard Async-MLA or Asynchronous Nesterov baselines, the incoming pseudo-gradients from the straggler nodes must contain distinct, conflicting geometric information that varies by tensor block<sup>3</sup> . This relies entirely on the selection of a pre-training corpus capable of inducing representation drift. Representation drift occurs when prolonged local optimization on highly skewed, non-IID data causes the embedding and representation spaces of the local workers to diverge semantically, producing parameter deltas that are internally coherent but globally conflicting<sup>8</sup> . Therefore, the defining requirement of the selected dataset is its structural capacity to be deterministically or probabilistically partitioned into severely non-IID shards that stress different functional blocks of the Transformer architecture. 

## **3. Exhaustive Analysis of Candidate Datasets for Non-IID Partitioning** 

To induce the requisite directional conflict in the pseudo-gradients, the underlying dataset must possess natural, easily isolatable sub-distributions. The original draft suggests a multilingual subset of the C4 dataset, but modern open-source data curation provides several alternative paradigms that may better serve the specific mechanical requirements of the Block-MLA experiment<sup>3</sup> . The following corpora represent the most structurally appropriate datasets, analyzed through the lens of decentralized, non-IID gradient behavior. 

### **3.1 The Multilingual Paradigm: The mC4 Corpus** 

The Colossal Clean Crawled Corpus (C4), originally developed by Google for the T5 models, includes a massive multilingual extension known as mC4<sup>21</sup> . The mC4 dataset comprises approximately 9.7 to 15.4 Terabytes of uncompressed text, utilizing the CLD3 language identification model to categorize scraped web content into 108 distinct linguistic subsets<sup>21</sup> . The distribution of data within mC4 is highly skewed, reflecting the natural occurrence of languages on the internet and providing a massive repository for simulating federated language modeling<sup>26</sup> . The structural metadata of the top languages provides a clear avenue for deterministic non-IID partitioning. 

|**Language Subset**|**Approximate Size**|**Estimated Token**<br>**Count (Billions)**|**Non-IID**<br>**Characteristics**|
|---|---|---|---|
|English (en)|10,401 GB|~2,500+ B|Baseline<br>high-resource Latin<br>script.|



|Russian (ru)|3,615 GB|~800+ B|Cyrillic script,<br>distinct subword<br>token boundaries.|
|---|---|---|---|
|Spanish (es)|1,613 GB|~380+ B|Latin script, heavy<br>morphological<br>inflection.|
|German (de)|1,404 GB|~330+ B|Latin script,<br>complex compound<br>noun structures.|
|Japanese (ja)|821 GB|~164 B|Logographic and<br>syllabic scripts<br>(Kanji, Hiragana).|
|Italian (it)|590 GB|~130 B|Latin script, highly<br>curated in specific<br>subsets.|



Partitioning mC4 by language guarantees an extreme form of non-IID data distribution. In the context of the Block-MLA simulation, Worker 1 (operating at the fast 1-second pace) could be assigned the English corpus, while Worker 5 (a straggler operating at the 15-second pace) could be assigned the Japanese or Russian corpus. Their local optimization trajectories will explore entirely different regions of the parameter space<sup>3</sup> . 

Multilingual training naturally induces extreme variance in the token embedding layer and the language-modeling classification head, as different languages rely on non-overlapping subsets of the Byte-Pair Encoding (BPE) subword vocabulary<sup>28</sup> . For example, a tokenizer trained primarily on Latin scripts will shatter Devanagari or Cyrillic characters into individual bytes, drastically altering the semantic density of the sequence and causing massive, isolated gradient updates in specific rows of the embedding matrix<sup>31</sup> . 

If Block-MLA is evaluated on mC4, the tensor-aware extrapolation coefficients will dynamically observe that the embedding matrix block from the Russian straggler conflicts heavily with the English-dominated global momentum. Block-MLA can then selectively apply aggressive look-ahead penalties to the embedding tensor while permitting more generalized representations in the deeper attention layers to integrate smoothly. Furthermore, evaluating on mC4 enables the measurement of "domain forgetting," demonstrating whether Block-MLA can prevent the fast English worker from catastrophically overwriting the linguistic representations of the straggler languages<sup>3</sup> . 

### **3.2 The Domain-Specific Paradigm: SlimPajama and RedPajama** 

An alternative to cross-lingual non-IID partitioning is cross-domain non-IID partitioning in a monolingual setting. SlimPajama is a meticulously deduplicated, multi-source dataset comprising 627 billion tokens, refined via MinHashLSH deduplication from the expansive 1.2 trillion token RedPajama dataset<sup>33</sup> . 

SlimPajama is structurally organized by distinct data domains, making it an ideal candidate for deterministic non-IID sharding that stresses reasoning and syntactic network components rather than just vocabulary embeddings. 

|**SlimPajama**<br>**Domain**|**Token Proportion**|**Total Tokens**|**Cognitive/Syntacti**<br>**c Profile**|
|---|---|---|---|
|CommonCrawl|52.2%|327 Billion|General web text,<br>high variance,<br>conversational<br>structure.|
|C4|26.7%|167 Billion|Filtered web text,<br>high baseline<br>quality.|
|GitHub|5.2%|32 Billion|Strict<br>programmatic<br>syntax, high<br>non-alphanumeric<br>token density.|
|ArXiv|4.6%|28 Billion|Complex<br>mathematical<br>notation, LaTeX<br>structures, logical<br>flow.|
|Books|4.2%|26 Billion|Long-form<br>narrative, deep<br>contextual<br>dependencies.|
|Wikipedia|3.8%|23 Billion|Encyclopedic facts,<br>structured<br>informational|



||||hierarchies.|
|---|---|---|---|
|StackExchange|3.3%|20 Billion|Q&A formatting,|
||||technical reasoning<br>and instruction.|



Assigning distinct domains to the 5 simulated workers creates a highly realistic federated learning scenario<sup>36</sup> . For example, the simulation could assign CommonCrawl to the fast worker, while allocating GitHub, ArXiv, Books, and Wikipedia to the stragglers. The fast worker will constantly update the global model with general web text momentum. Meanwhile, the stragglers will submit pseudo-gradients optimized for structural programming logic or complex mathematical syntax. 

Research into cross-domain data mixtures indicates that the token distributions between these domains diverge sharply<sup>33</sup> . The Kullback-Leibler (KL) divergence of non-alphanumeric tokens, such as mathematical operators or code syntax markers, is vastly different in the ArXiv and GitHub datasets compared to CommonCrawl<sup>33</sup> . Because the syntax and structural logic of code differ fundamentally from general text, the local Stochastic Gradient Descent (SGD) steps will steer the attention matrices in conflicting directions<sup>12</sup> . 

Standard Async-MLA computes a single scalar alignment score for the entire model update, which might register as a net-negative alignment due to the sheer volume of syntactic differences, causing the entire code update to be penalized<sup>3</sup> . Block-MLA, computing alignment tensor-by-tensor, would mathematically recognize that while certain projection matrices conflict, lower-level syntactic attention heads remain highly aligned with general language structure, thereby preserving the underlying logical capabilities of the code shard<sup>3</sup> . 

### **3.3 The Educational Quality Paradigm: FineWeb and FineWeb-Edu** 

If the experimental design requires evaluating Block-MLA's ability to extract high-signal updates from a sea of low-quality momentum, the FineWeb and FineWeb-Edu datasets provide an optimal environment<sup>35</sup> . FineWeb is a colossal 15 trillion-token English web corpus distilled from 96 CommonCrawl snapshots utilizing aggressive repetition heuristics and URL blocklists<sup>35</sup> . FineWeb-Edu is a premium 1.3 trillion-token subset of FineWeb, aggressively filtered using a Llama-3-70B-Instruct classifier to retain only documents scoring high for educational value<sup>21</sup> . While FineWeb does not have native domain splits like SlimPajama, it can be utilized to simulate heterogeneous data quality. In a decentralized training run, it is highly probable that not all edge nodes have access to highly curated data<sup>8</sup> . The simulation could assign the premium FineWeb-Edu dataset to the slow straggler workers, while the fast worker trains on the raw, unfiltered FineWeb corpus<sup>39</sup> . 

This architecture creates a fascinating optimization dynamic. The high-frequency updates from Worker 1 will dominate the global momentum with noisy, lower-quality trajectories. The low-frequency, stale updates from the stragglers will contain high-quality, dense reasoning 

signals<sup>39</sup> . If standard asynchronous methods blindly aggregate these updates, the high-quality geometric signals of the stale updates will be geometrically crushed by the fast worker's dominant momentum. Block-MLA's ability to protect and integrate the high-quality geometric signals of the educational stale updates would serve as a powerful proof of concept, directly translating to improved performance on downstream reasoning benchmarks like MMLU and ARC<sup>31</sup> . 

### **3.4 The Syntactic and Programmatic Paradigm: StarCoderData** 

For simulations specifically targeting reasoning capabilities and algorithmic structural retention, StarCoderData provides a highly focused corpus. Derived from The Stack, StarCoderData contains 783 GB of curated code across 86 programming languages, yielding approximately 250 billion tokens<sup>42</sup> . 

This dataset allows for a non-IID split based entirely on programming paradigms rather than natural language. Worker 1 could be assigned Python, representing high-level, indentation-based syntax. Worker 2 could receive C++, introducing low-level, compiled memory-management syntax. Worker 3 could process JavaScript, highlighting asynchronous web patterns, while Worker 4 processes HTML/CSS markup, and Worker 5 handles Shell scripting<sup>44</sup> . 

Because programming languages share highly overlapping subword tokens (such as return, if, {, }, and ()) but utilize them in vastly different syntactic structures, the geometric conflict will predominantly manifest in the attention projection matrices and the deep MLP layers rather than in the vocabulary embedding tables<sup>13</sup> . This provides a highly specific testbed for Block-MLA's tensor-wise correction mechanism deep within the Transformer body, completely isolating the variables of syntactic logic from linguistic vocabulary. 

## **4. Mathematical Structuring of Non-IID Partitioning Strategies** 

Once the foundational dataset is selected, the mathematical method of distributing the data 

among the simulated workers governs the absolute difficulty of the outer optimization problem. 

### **4.1 Deterministic Sharding vs. Probabilistic Distribution** 

The most severe test for Block-MLA is strict deterministic sharding, where the intersection of data domains between workers is exactly zero. As outlined in Section 3.2, assigning exactly one SlimPajama domain to each worker guarantees maximum gradient misalignment. However, in real-world decentralized environments, data silos are rarely perfectly isolated. A federated hospital network contains varied patient demographics but shares foundational medical terminology across all nodes<sup>48</sup> . To accurately model real-world statistical heterogeneity and prevent the simulation from appearing artificially pathological, probabilistic distributions are required. 

### **4.2 Probabilistic Sharding via Dirichlet Distributions** 

In the academic literature surrounding Federated Learning and Distributed 

Low-Communication training, the gold standard for mathematically simulating non-IID client data is the Dirichlet distribution<sup>50</sup> . 

The Dirichlet distribution is a multivariate continuous probability distribution parameterized by a 

concentration vector , where each . It is utilized to allocate continuous proportions of the total dataset to different workers based on the underlying classes, languages, or domains<sup>53</sup> . The Probability Density Function (PDF) is formally defined as: 



where and represents the multivariate Beta function<sup>53</sup> . 

By varying the scalar concentration parameter , the Block-MLA simulation can smoothly interpolate between perfectly IID and extreme non-IID data regimes, allowing for a highly controlled analysis of gradient divergence<sup>54</sup> . 

|**Dirichlet Parameter (α)**|**Distribution**<br>**Characteristics**|**Optimization Impact**|
|---|---|---|
|**(or**<br>**)**|Near-uniform, IID split.|Each worker receives an<br>equal mix of all domains<br>(e.g., 20% Web, 20% Code,<br>20% Math). Minimal<br>representation drift.|
||Moderate Heterogeneity.|Workers receive skewed<br>proportions, but most<br>workers still process a<br>fraction of every domain.<br>Simulates standard<br>geographic data silos.|
||High Heterogeneity.|Noticeable imbalance.<br>Often used as the standard<br>benchmark for non-IID<br>federated algorithms.|





Pathological Heterogeneity. The distribution becomes sparse. A worker may receive 99% of one domain and 1% of another. Maximizes directional conflict. 

The hypothesis stated in the Block-MLA draft—that standard Momentum Look-Ahead will converge but suffer accuracy degradation due to geometric conflict—will be most visible and 

measurable in the regime<sup>3</sup> . Under this pathological regime, the global momentum 

will be entirely dominated by the fast worker's specific data domain due to its high update frequency. When a stale pseudo-gradient arrives from a slow worker featuring a highly isolated, distinct data domain, the normalized block compatibility score will 

frequently register as strongly negative ( ), triggering Block-MLA's aggressive extrapolation penalty to pull the block back toward the global trajectory<sup>3</sup> . 

Establishing experiments across , , and will comprehensively prove that Block-MLA adapts dynamically to the level of heterogeneity, ensuring it does not degrade performance in IID settings while dominating baselines in non-IID settings<sup>52</sup> . 

## **5. Technical Implementation and System Engineering** 

Constructing the single-GPU simulated environment for Block-MLA requires careful orchestration of data streaming, tokenization, and optimizer engineering. Because simulating asynchronous distributed training on a single device involves managing multiple independent model states, inner optimizers, and global communication buffers, VRAM memory efficiency and computational flow are paramount. 

### **5.1 Dataset Streaming and Memory Constraints** 

Training a NanoGPT model (ranging from 15M to 90M parameters) across 5 simulated workers requires maintaining 5 distinct local model replicas, 1 global model, and their respective AdamW optimizer momentum and variance states in the GPU VRAM<sup>3</sup> . Loading a massive dataset like SlimPajama (627B tokens) or FineWeb-Edu into local memory is computationally impossible and practically unnecessary. 

The experiment must utilize the datasets library from HuggingFace, specifically leveraging the streaming=True parameter to instantiate IterableDatasets<sup>60</sup> . 

Using iterable datasets allows the simulator to dynamically fetch micro-batches for each worker's inner loop directly over the network or from disk shards without exhausting system RAM<sup>60</sup> . Furthermore, stateful asynchronous data loaders must be engineered so that when the 

fast worker (Worker 1) executes 15 inner steps in the exact time Worker 2 executes 1, Worker 1 continuously draws new data sequentially without resetting its stream or causing blocking I/O operations that would stall the simulation<sup>60</sup> . 

### **5.2 Tokenization and the Embedding Layer Dilemma** 

Tokenization plays a critical, often overlooked role in non-IID distributed training. If the workers are training on divergent domains, the specific distribution of tokens they encounter and update will vary drastically<sup>29</sup> . The GPT-2 Byte-Pair Encoding (BPE) tokenizer is standard for NanoGPT implementations and is recommended for this setup<sup>32</sup> . 

However, the token embedding layer presents a unique optimization vulnerability under the Block-MLA algorithm. The embedding matrix is essentially a highly sparse lookup table; its rows are only updated if the corresponding tokens appear in the current micro-batch<sup>29</sup> . Under non-IID conditions, Worker 1 (processing web text) will frequently update common linguistic tokens, while Worker 3 (processing code) will heavily update syntax tokens<sup>31</sup> . 

When Worker 3 submits its stale pseudo-gradient, massive portions of its embedding gradient 

matrix will be exactly zero. If Block-MLA computes the cosine similarity globally across the entire embedding tensor, the extreme sparsity of the update will artificially distort 

the alignment score, leading to erratic coefficients. Advanced optimizer research, such as the Muon optimizer utilized in NanoGPT speedruns and the TailOPT heavy-tailed gradient framework, explicitly warns against applying 

matrix-structured orthogonalization or complex momentum corrections to -D parameters and embedding layers<sup>17</sup> . To ensure stability, Block-MLA's implementation should filter parameters using PyTorch's named_parameters() function. This allows the simulation to 

selectively apply the Block-MLA modulation exclusively to the 2D weight matrices (attention projections and MLPs) while defaulting to standard momentum or decoupled AdamW updates for 1D biases, LayerNorm weights, and the sparse vocabulary embedding matrix<sup>38</sup> . 

### **5.3 Engineering the Tensor-Aware Outer Optimizer** 

The core innovation of Block-MLA is the mathematical transition from a global momentum 

coefficient to a dynamic, tensor-block specific coefficient 3. The outer optimizer must be implemented as a custom, highly optimized PyTorch Optimizer subclass to prevent the global synchronization step from becoming a computational bottleneck. 

During the global aggregation step, the server receives the pseudo-gradient from a worker. 

The optimizer must iterate through each eligible parameter block : 

1. **Retrieve State:** Extract the global momentum block and the incoming pseudo-gradient block . 

2. **Normalization:** Compute the L2 normalized vectors and . 

3. **Cosine Similarity:** Calculate the directional compatibility via a highly optimized dot product<sup>3</sup> . 

4. **Coefficient Modulation:** Calculate the block-specific based on the value of . If , the penalty increases dynamically, pulling the conflicting block back toward the 

established global momentum trajectory. 

5. **State and Parameter Update:** Execute the reparameterized block updates efficiently using PyTorch's add_ and mul_ in-place operations to conserve VRAM: 





This block-wise operation preserves the directional precision necessary for non-IID training but avoids the severe computational overhead of explicit Singular Value Decomposition (SVD) or geometric orthogonal projection utilized in methods like HeLoCo or Muon<sup>3</sup> . By relying on simple dot products and scalar modulation per tensor, Block-MLA maintains exceptional throughput. 

## **6. Measuring Success: Metrics and Analytical Framework** 

The simulated experiment must track specific, granular metrics to conclusively prove that Block-MLA successfully navigates the trade-offs of the asynchronous, non-IID environment better than existing baselines<sup>3</sup> . 

### **6.1 Staleness Robustness and Domain Forgetting** 

In the pace configuration, Worker 1 executes 15 outer steps for every 1 step executed by the stragglers. The primary failure mode of standard Asynchronous Nesterov optimization in this environment is "domain forgetting"—the global model continually optimizes for Worker 1's dataset and effectively ignores or is actively destabilized by the infrequent updates from Workers 2 through 5<sup>3</sup> . 

To mathematically measure this resilience, the experiment must evaluate the global model against a segmented validation set. If utilizing SlimPajama, the validation set must be strictly partitioned by domain. A successful Block-MLA run will demonstrate that the validation loss on Worker 2's specific domain (e.g., Code or ArXiv) decreases monotonically upon the integration 

of Worker 2's highly stale pseudo-gradient<sup>69</sup> . In contrast, standard Async-MLA will likely exhibit a validation loss spike across all domains when a stale gradient is integrated, due to the uniform application of geometric conflict penalties across all layers<sup>3</sup> . 

### **6.2 Tracking Representation Drift via Downstream Baselines** 

Representation drift is a severe pathology where the internal features learned by local workers diverge so significantly that their aggregate average becomes semantically incoherent<sup>15</sup> . Recent empirical studies on DiLoCo architectures have shown that while pretraining loss may appear stable, representation drift can completely collapse downstream performance on reasoning benchmarks like MMLU, GSM8K, and HumanEval<sup>8</sup> . Block-MLA inherently combats this drift by enforcing momentum alignment strictly at the block level, preventing local attention matrices from straying into incompatible loss basins. 

To quantify this, the simulator must periodically measure the pairwise cosine similarity of the 

pseudo-gradients and before they are integrated by the global outer optimizer. Over the course of the 300 outer steps (simulating 24,000 inner steps), standard DiLoCo baselines will show rapidly decreasing cosine similarity between workers as they drift<sup>3</sup> . Block-MLA's penalty mechanism should exert a strong regularizing force, keeping the geometric divergence bounded. Furthermore, validating the resulting model checkpoints on MMLU and GSM8K will prove that Block-MLA preserves the necessary feature geometry for downstream reasoning, directly addressing the core vulnerability of asynchronous distributed training<sup>15</sup> . 

### **6.3 Fixed Time Budget vs. Fixed Token Budget** 

As explicitly requested in the draft methodology, the final results must be evaluated through two distinct analytical lenses: Loss vs. Steps (Fixed Token Budget) and Loss vs. Wall-clock time (Fixed Time Budget)<sup>3</sup> . 

|**Evaluation Lens**|**Theoretical Expectation**|**Proof of Block-MLA**<br>**Efficacy**|
|---|---|---|
|**Fixed Token Budget (Loss**<br>**vs. Steps)**|Synchronous Nesterov<br>(standard DiLoCo) provides<br>the theoretical lower bound<br>for loss, as it suffers zero<br>staleness. All asynchronous<br>methods incur a step<br>penalty here.|Block-MLA must<br>demonstrate that its loss<br>trajectory remains<br>significantly closer to the<br>Synchronous Nesterov<br>bound than standard<br>Async-MLA or naive Async<br>Nesterov, proving superior<br>gradient utilization.|
|**Fixed Time Budget (Loss**|Synchronous Nesterov|Block-MLA must achieve|



|**vs. Time)**|performs poorly, as it is<br>bottlenecked, waiting 15<br>seconds for stragglers to<br>finish every step<sup>3</sup>.|the absolute lowest total<br>validation loss within a set<br>time frame (e.g., 2 hours of<br>simulated wall-clock time),<br>i|
|---|---|---|
||Asynchronous methods<br>advance the global step|confirming that its dynamic<br>tensor corrections safely|
||counter exponentially|harness the raw speed of|
||faster.|asynchrony without<br>sacrificing model quality.|



## **7. Strategic Recommendations and Experimental Blueprint** 

The transition from synchronous, datacenter-bound LLM training to decentralized, asynchronous low-communication environments represents a critical frontier in AI scaling<sup>1</sup> . However, the pathologies of gradient staleness and representation drift under non-IID conditions threaten the mathematical viability of these architectures<sup>8</sup> . The Block-MLA algorithm provides a theoretically sound, computationally lightweight mechanism to resolve these directional conflicts at the tensor-block level, bypassing the need for expensive geometric projections<sup>3</sup> . 

To rigorously prove the claims outlined in the Block-MLA draft and ensure the resulting publication withstands peer review, the experimental design must adopt the following finalized specifications: 

1. **Primary Dataset Selection:** Transition the experiment from the proposed mC4 subset to the **SlimPajama (627B)** corpus. While mC4 provides linguistic heterogeneity, cross-lingual training introduces extreme vocabulary sparsity that complicates the measurement of geometric alignment in deeper network layers. SlimPajama's native domain splits (CommonCrawl, GitHub, ArXiv, Books) guarantee robust structural divergence while maintaining a shared English vocabulary space, allowing Block-MLA's tensor corrections in the attention and MLP layers to be isolated and evaluated accurately. 

2. **Mathematical Partitioning:** Utilize a **Dirichlet distribution ( )** to assign dataset proportions to the 5 simulated workers. This ensures severe, realistic data heterogeneity, forcing the fast worker and the straggler workers to optimize along vastly different trajectories, thereby triggering the exact geometric conflicts Block-MLA is designed to solve. 

3. **System Engineering:** Leverage HuggingFace load_dataset with streaming=True to maintain the memory efficiency of the single-GPU simulation. Crucially, the custom Block-MLA PyTorch optimizer must utilize named_parameters() to filter out 1D tensors 

and the highly sparse token embedding matrix from the dynamic extrapolation. This 

ensures that vocabulary sparsity does not corrupt the block-wise cosine similarity metrics. 

4. **Analytical Focus:** Prioritize the tracking of domain-specific validation loss spikes upon the integration of the extreme stragglers (the 15-second pace nodes). Furthermore, tracking downstream performance on MMLU and GSM8K will definitively prove that Block-MLA prevents the catastrophic representation drift that currently plagues asynchronous DiLoCo pipelines. 

By structuring the experimental framework around these precise data, mathematical, and architectural parameters, the resulting analysis will provide incontrovertible empirical evidence of Block-MLA's capacity to stabilize and accelerate asynchronous, decentralized LLM optimization. 

#### **Works cited** 

1. DiLoCo: Distributed Low-Communication Training of Language, tps://arxiv.org/html/2311.08105v2 

   - <u>https://arxiv.org/html/2311.08105v2</u> 

2. GPU Networking for AI Clusters: InfiniBand vs RoCE vs Spectrum-X, <u>https://www.spheron.network/blog/gpu-networking-infiniband-roce-spectrum-xguide/</u> 

3. Block-MLA.docx 

4. Training Neural Networks at Any Scale - arXiv, <u>https://arxiv.org/pdf/2511.11163</u> 

5. (PDF) Distributed Low-Communication Training with Decoupled, <u>https://www.researchgate.net/publication/396250343_Distributed_Low-Communi cation_Training_with_Decoupled_Momentum_Optimization</u> 

6. MAGNET: Autonomous Expert Model Generation via Decentralized, <u>https://arxiv.org/html/2603.25813v1</u> 

7. EFFICIENT DISTRIBUTED OPTIMIZATION UNDER HEAVY-TAILED, <u>https://openreview.net/pdf?id=HvLbOMgIax</u> 

8. What happens when nanochat meets DiLoCo? - arXiv, <u>https://arxiv.org/pdf/2511.13761</u> 

9. HeLoCo: Efficient asynchronous low-communication training under, tps://arxiv.org/html/2606.00271v1 

   - <u>https://arxiv.org/html/2606.00271v1</u> 

10. DiLoCo - Distributed Low-Communication Training of Language, <u>https://www.scribd.com/document/1056963674/DiLoCo-Distributed-Low-Comm unication-Training-of-Language-Models</u> 

11. FL Vs Distributed Training Position | PDF | Data | Communication, <u>https://www.scribd.com/document/1064084737/FL-vs-Distributed-Training-Positi on</u> 

12. Daily Papers - Hugging Face, https://huggingface.co/papers?q=sparse%20RoPE 

13. A Brief Summary of Optimization in Deep Learning in New Era, <u>https://www.researchgate.net/publication/405297794_A_Brief_Summary_of_Opti mization_in_Deep_Learning_in_New_Era</u> 

14. 

15. 1 Introduction - arXiv, https://arxiv.org/html/2511.13761v1 

16. DePIN's Imperfect Present & Promising Future: A Deep Dive, 

<u>https://www.compound.vc/writing/depin</u> 

17. Efficient Distributed Optimization under Heavy-Tailed Noise - arXiv, <u>https://arxiv.org/pdf/2502.04164</u> 

18. [2511.13761] What happens when nanochat meets DiLoCo? - arXiv, <u>https://arxiv.org/abs/2511.13761</u> 

19. A Guide to Implementing and Training Generative ... - ROCm™ Blogs, <u>https://rocm.blogs.amd.com/artificial-intelligence/nanoGPT-JAX/README.html</u> 

20. (PDF) What happens when nanochat meets DiLoCo? - ResearchGate, <u>https://www.researchgate.net/publication/397740030_What_happens_when_nan ochat_meets_DiLoCo</u> 

21. oumi.datasets.pretraining, tps://oumi.ai/docs/en/latest/api/oumi.datasets.pretraining.html 

   - <u>https://oumi.ai/docs/en/latest/api/oumi.datasets.pretraining.html</u> 

22. google/mt5-large at 06cc97cf7b209d71d3cf17c984bd9928f784bb6c, <u>https://huggingface.co/google/mt5-large/blame/06cc97cf7b209d71d3cf17c984b d9928f784bb6c/README.md</u> 

23. allenai/c4 at mC4_3.1.0 - Hugging Face, tps://huggingface.co/datasets/allenai/c4/tree/mC4_3.1.0/multilingual 

   - <u>https://huggingface.co/datasets/allenai/c4/tree/mC4_3.1.0/multilingual</u> 

24. gsarti/clean_mc4_it · Datasets at Hugging Face, <u>https://huggingface.co/datasets/gsarti/clean_mc4_it</u> 

25. bertin-project/mc4-sampling at aa99455 - Dataset files - Hugging Face, <u>https://huggingface.co/datasets/bertin-project/mc4-sampling/commit/aa994555f 6d6bc8a7b342dbd23cb51c09700448b</u> 

26. The C4 Multilingual Dataset #5265 - allenai allennlp - GitHub, <u>https://github.com/allenai/allennlp/discussions/5265</u> 

27. FedNLP: Benchmarking Federated Learning Methods for Natural, <u>https://aclanthology.org/2022.findings-naacl.13.pdf</u> 

28. DEPT: Decoupled Embeddings for Pre-training Language Models, <u>https://openreview.net/forum?id=vf5aUZT0Fz</u> 

29. Mitigating Gradient Inversion Risks in Language Models via Token, <u>https://arxiv.org/pdf/2602.15897</u> 

30. Vector researchers advance representation learning and deep, <u>https://vectorinstitute.ai/vector-researchers-advance-representation-learning-an d-deep-learning-research-at-iclr-2026/</u> tps://huggingface.co/papers?q=m-MMLU 

31. Daily Papers - Hugging Face, https://huggingface.co/papers?q=m-MMLU 

32. sajalregmi4/arkios-1b-base - Hugging Face, <u>https://huggingface.co/sajalregmi4/arkios-1b-base</u> 

33. SlimPajama-DC: Understanding Data Combinations for LLM Training, <u>https://ar5iv.labs.arxiv.org/html/2309.10818</u> 

34. SlimPajama-DC: Understanding Data Combinations for LLM Training, <u>https://arxiv.org/html/2309.10818v3</u> 

35. Hugging Face Datasets Guide - Department of Computer Science, <u>https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/datasets202 5.html</u> 

36. DKYoon/SlimPajama-6B · Datasets at Hugging Face, <u>https://huggingface.co/datasets/DKYoon/SlimPajama-6B</u> 

37. MULTI-AGENT COLLABORATIVE DATA SELECTION - OpenReview, <u>https://openreview.net/notes/edits/attachment?id=O5NAVkTHWf&name=pdf</u> 

38. Learned Subspace Compression for Communication ... - OpenReview, <u>https://openreview.net/attachment?id=9ztCmwxYxN&name=pdf</u> 

39. FineWeb: HuggingFace's 15-trillion-token web corpus - ZeroEntropy, <u>https://zeroentropy.dev/concepts/fineweb/</u> 

40. HuggingFaceFW/fineweb-edu · Datasets at Hugging Face, <u>https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu</u> 

41. SmolLM3: smol, multilingual, long-context reasoner - Hugging Face, <u>https://huggingface.co/blog/smollm3</u> 

42. bigcode/starcoderdata · Datasets at Hugging Face, tps://huggingface.co/datasets/bigcode/starcoderdata 

   - <u>https://huggingface.co/datasets/bigcode/starcoderdata</u> 

43. A Survey of Neural Code Intelligence: Paradigms, Advances ... - arXiv, <u>https://arxiv.org/html/2403.14734v3</u> 

44. paper_final.md · StentorLabs/Portimbria-150M-Vs.-SmolLM2-135M, <u>https://huggingface.co/datasets/StentorLabs/Portimbria-150M-Vs.-SmolLM2-135 M/blob/main/paper_final.md</u> 

45. Subword Tokenizer-Free Generative LLMs via Sparse, <u>https://aclanthology.org/2024.emnlp-main.1217.pdf</u> 

46. In the long (context) run - Harm de Vries, <u>https://www.harmdevries.com/post/context-length/</u> 

47. Prepare 1T tokens - Docs - Lightning AI, tps://lightning.ai/docs/platorm/data/prepf f 

- <u>https://lightning.ai/docs/platorm/data/prepf</u> -data/prepare-1-trillion-token-dataset 

- 48. FedNP: Towards Non-IID Federated Learning via Federated Neural, <u>https://ojs.aaai.org/index.php/AAAI/article/view/26237/26009</u> 

49. (PDF) Handling Non-IID Data in Federated Learning - ResearchGate, <u>https://www.researchgate.net/publication/375075344_Handling_Non-IID_Data_in_ Federated_Learning_An_Experimental_Evaluation_Towards_Unified_Metrics</u> 

50. Addressing Non-IID with Data Quantity Skew in Federated Learning, <u>https://www.mdpi.com/2078-2489/16/10/861</u> 

51. partition - FedLab 1.3.0 documentation, <u>https://fedlab.readthedocs.io/en/master/autoapi/fedlab/utils/dataset/partition/inde x.html</u> 

52. Federated Learning: Distributed AI in the Age | MI - 超智諮詢 , tps://www.meta-intelligence.tech/en/insight-federated-learning 

   - <u>https://www.meta-intelligence.tech/en/insight-federated-learning</u> 

53. Dirichlet Distribution - GeeksforGeeks, <u>https://www.geeksforgeeks.org/machine-learning/dirichlet-distribution/</u> 

54. Sampling Consistent and Accurate Contribution Values in Federated, <u>https://arxiv.org/html/2602.05693v1</u> 

55. Data-Free Black-Box Federated Learning via Zeroth-Order Gradient, <u>https://ojs.aaai.org/index.php/AAAI/article/view/34126/36281</u> 

56. Xtra-Computing/NIID-Bench: Federated Learning Benchmark - GitHub, <u>https://github.com/Xtra-Computing/NIID-Bench</u> 

57. A Federated Fine-Tuning Framework for Large Language Models, <u>https://www.mdpi.com/2227-7390/13/19/3201</u> 

58. Decentralized Federated Learning with Non-IID Data - ResearchGate, <u>https://www.researchgate.net/publication/398885479_Decentralized_Federated_L earning_with_Non-IID_Data_Challenges_Trends_and_Future_Opportunities</u> 

59. On Provable Benefits of Muon in Federated Learning | alphaXiv, <u>https://www.alphaxiv.org/abs/2510.03866</u> 

60. Tokenization at Scale | Jaxformer: Scaling Modern Transformers, <u>https://jaxformer.com/tokenization/</u> tps://docs.lancedb.com/datasets/fineweb-edu ineweb-edu 

61. FineWeb-Edu - LanceDB, https://docs.lancedb.com/datasets/fineweb-edu 

62. Create sft.py · ericflo/Llama-3.1-8B-ContinuedTraining at 86b5e8a, <u>https://huggingface.co/ericflo/Llama-3.1-8B-ContinuedTraining/commit/86b5e8a9 c1cc225ec2af840df</u> **f** <u>8724543040875</u> 

63. Decepticons: Corrupted Transformers Breach Privacy in Federated, <u>https://www.researchgate.net/publication/358260318_Decepticons_Corrupted_Tr ansformers_Breach_Privacy_in_Federated_Learning_for_Language_Models</u> 

64. fineweb.py - karpathy/build-nanogpt - GitHub, tps://github.com/karpathy/build-nanogpt/blob/master/fineweb.py ineweb.py 

   - <u>https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py</u> 

65. 6ae86d05-5cb2-4e40-a512-63246fd08e45.txt - GitHub, <u>https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_s hort/2025-05-25_EvenFasterReduce/6ae86d05-5cb2-4e40-a512-63246fd08e45. txt</u> 

66. Reproducing the nanoGPT Speedrun: What Actually Moves the Loss, tps://frontiercheckpoint.com/reproductions/reproducing-nanogpt-speedrun/ 

   - <u>https://frontiercheckpoint.com/reproductions/reproducing-nanogpt-speedrun/</u> 

67. Towards Execution-Grounded Automated AI Research - arXiv, <u>https://arxiv.org/html/2601.14525v1</u> 

68. artifacts/muon_wdsched_cmpatino-1 ... - Hugging Face, <u>https://huggingface.co/buckets/ml-intern-explorers/efficient-optimizer-collab/tre e/artifacts/muon_wdsched_cmpatino-1/train_gpt_simple.py</u> 

69. DataFlex: A Unified Framework for Data-Centric Dynamic Training of, tps://www.alphaxiv.org/abs/2603.26164 

- <u>https://www.alphaxiv.org/abs/2603.26164</u> 

- 70. Efficient Distributed Optimization under Heavy-Tailed Noise, <u>https://icml.cc/virtual/2025/poster/46381</u> 

71. FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs, <u>https://arxiv.org/pdf/2606.19025</u> 

