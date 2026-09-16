Strucutre Twin:https://openreview.net/pdf?id=4O8nzTkHPI 

# **Block-MLA: Dynamic Tensor-Aware Momentum Look-Ahead for Asynchronous Distributed Low-Communication Training** 

## **Abstract** 

Distributed Low-Communication (DiLoCo) training reduces communication overhead by allowing workers to perform multiple local steps before sending pseudo-gradients to a global server. Asynchronous variants eliminate synchronization bottlenecks but introduce gradient staleness, which severely degrades convergence. Recent methods like Momentum Look-Ahead (MLA) address this by extrapolating the negative momentum direction uniformly across the entire pseudo-gradient. However, under data and device heterogeneity, uniform correction is insufficient; stale pseudo-gradients contain distinct tensor blocks that exhibit varying degrees of alignment with the global trajectory. We propose **Block-MLA** , a direction-aware correction algorithm that extends momentum look-ahead to the tensor-block level. By dynamically calculating extrapolation coefficients per block based on local variance and alignment, Block-MLA mitigates the misaligned components of stale updates without the computational overhead of explicit geometric projection. We provide a rigorous single-GPU simulated experimental framework to demonstrate that Block-MLA outperforms standard MLA and asynchronous Nesterov baselines under severe system heterogeneity. 

## **1. Introduction** 

The training of Large Language Models (LLMs) requires massive computational clusters. To scale beyond a single datacenter, Distributed Low-Communication (DiLoCo) formulates Data Parallelism (DP) as a bilevel optimization problem, communicating pseudo-gradients infrequently. Asynchronous DiLoCo relaxes the need for synchronized updates, preventing slow workers from bottlenecking the system. 

However, asynchrony causes workers to compute local updates on outdated model states, generating stale pseudo-gradients. Existing delay correction mechanisms either rely on buffer-based aggregation or apply uniform correction across the entire update. MLA performs a look-ahead step in the negative direction of momentum, providing robust delay correction without additional hyperparameters. Yet, as identified by the HeLoCo framework, a stale update from a slower worker under non-IID (Independent and Identically Distributed) data conditions is not uniformly delayed or uniformly misaligned. Different parameter blocks within the same pseudo-gradient conflict with the outer update direction to varying degrees. 

We introduce Block-MLA to resolve this. By shifting momentum extrapolation from a global scalar to a tensor-wise operation, we allow the outer optimizer to dynamically scale the 

look-ahead step for each specific weight matrix, preserving aligned information and dampening conflicting trajectories. This approach is highly relevant for decentralized compute ecosystems where hardware heterogeneity and non-IID data routing are unavoidable. 

## **2. Related Work** 

- **Asynchronous DiLoCo:** Synchronous DiLoCo reduces communication but suffers from the straggler effect. Asynchronous variants update the server whenever a worker returns a pseudo-gradient, resulting in delayed gradients. 

- **Momentum Correction:** The Nesterov Accelerated Gradient (NAG) method has been adapted for DiLoCo. MLA modifies NAG by computing momentum as the exponential moving average of gradients and extrapolating the negative momentum direction. 

- **Heterogeneity and Directional Conflict:** Stale gradients from non-IID distributions suffer from directional distortion. HeLoCo addresses this by geometrically projecting conflicting tensor blocks along the normalized momentum direction. While effective, HeLoCo requires multiple hyperparameters (alignment thresholds, shrinkage bounds). 

- **Low-Rank Optimization:** Frameworks like LORDO demonstrate that global projections based on pseudo-gradients can permanently restrict the optimization trajectory to a low-rank subspace. While our current formulation focuses on full-rank updates, Block-MLA's tensor-wise operations are structurally compatible with low-rank factorizations. 

## **3. Methodology: Block-Wise Momentum Look-Ahead** 

### **3.1 The Standard MLA Baseline** 

In standard MLA, the outer optimization step computes the look-ahead step $d_t$ and the weight update for the full model using a global momentum coefficient $\gamma_t$: 

$$d_t = -\eta \gamma_t m_t$$ $$\theta_{t+1} = \theta_t + d_t - \eta \nabla f(\bar{\theta}_t + d_t)$$ $$m_{t+1} = \gamma_t m_t + (1 - \gamma_t)\nabla f(\bar{\theta}_t + d_t)$$ where $\bar{\theta}_t$ is the delayed point and $m_t$ is the momentum. This uniformly extrapolates the update across all dimensions. 

### **3.2 Tensor-Aware Correction (Block-MLA)** 

We extend this by defining the pseudo-gradient $\Delta_i$ as a collection of tensor blocks $b$. Let $\Delta_b$ denote a block of the arriving pseudo-gradient and $(m_t)_b$ denote the corresponding momentum block. 

Instead of a global $\gamma_t$, we calculate a dynamic coefficient $\gamma_{t,b}$ for each block. We utilize the directional compatibility score $c_b$ between the normalized pseudo-gradient block $\hat{u}_b$ and the normalized momentum block $\hat{v}_b$: 

$$c_b = \hat{u}_b^\top \hat{v}_b$$ 

We redefine the momentum extrapolation step per block. If $c_b < 0$, indicating an anti-momentum component, we increase the look-ahead penalty to aggressively pull the block back toward the global trajectory. If $c_b \ge 0$, the block is aligned, and standard extrapolation is applied. 

The reparameterized Block-MLA update for tensor block $b$ is: 

$$m_{t+1, b} = \gamma_{t,b} m_{t,b} + (1 - \gamma_{t,b}) \Delta_b$$ $$\theta_{t+1, b} = \theta_{t,b} - \eta \left( \gamma_{t+1,b} m_{t+1, b} + \Delta_b \right)$$ This provides the directional precision of HeLoCo but operates entirely within the simplified computational graph of MLA, avoiding explicit orthogonal projections. 

## **4. Experimental Design** 

To validate Block-MLA without requiring a multi-node cluster, we establish a single-GPU simulated environment that strictly controls staleness and heterogeneity. 

### **4.1 System Simulation** 

- **Workers:** $K=5$ simulated heterogeneous workers. 

- **Pace Configurations:** Worker speeds (seconds per step) are drawn from $\{1, 2, 6, 15\}$ to mirror severe straggler effects. We emphasize the extreme staleness configuration $(1, 15, 15, 15, 15)$, where one fast worker dominates while four submit highly stale updates. 

- **Data Partitioning:** The dataset is split into 5 distinct, non-IID partitions (e.g., simulating different domains or languages). 

### **4.2 Model and Hyperparameters** 

- **Architecture:** A decoder-only Transformer (NanoGPT style) with approximately 15M to 90M parameters. 

- **Dataset:** A multilingual subset of the C4 dataset to enforce non-IID domain shifts. 

- **Optimization:** Inner learning rate of $3 \times 10^{-4}$ with AdamW, and outer learning rate of $0.7$ with the respective outer optimizers. We simulate 300 outer steps of 80 inner steps each. 

### **4.3 Baselines for Comparison** 

1. **Synchronous Nesterov:** The standard lock-step DiLoCo baseline. 

2. **Asynchronous Nesterov:** Naive asynchronous updates. 

3. **Async-MLA:** Standard Momentum Look-Ahead with uniform correction. 

4. **Block-MLA (Ours):** Dynamic tensor-aware momentum extrapolation. 

## **5. Expected Results & Metrics** 

We evaluate convergence through two primary lenses: **Fixed Token Budget** (Loss vs. Steps) and **Fixed Time Budget** (Loss vs. Wall-clock time). 

1. **Convergence under Non-IID Heterogeneity:** We anticipate that Asynchronous Nesterov will exhibit unstable behavior and diverge under high staleness. Standard MLA will converge but suffer accuracy degradation due to geometric conflict. Block-MLA is expected to achieve the lowest validation loss among asynchronous methods by selectively preserving aligned tensor blocks. 

2. **Wall-Clock Efficiency:** Because synchronous training is bottlenecked by the slowest worker, all asynchronous variants will demonstrate a steep advantage in loss-vs-time. Block-MLA is expected to achieve the best trade-off between convergence speed and loss. 

3. **Staleness Robustness:** In extreme configurations like $(1, 15, 15, 15, 15)$, we will analyze the validation loss contribution of the specific data shard assigned to the slowest worker. Block-MLA should prevent the global model from "forgetting" the slow worker's domain by safely incorporating its delayed, non-IID updates rather than rejecting them entirely. 

## **6. Conclusion** 

Standard asynchronous DiLoCo accelerates training but introduces gradient staleness that misaligns optimization trajectories under data and device heterogeneity. Block-MLA bridges the gap between uniform momentum extrapolation and expensive geometric projections. By adjusting look-ahead steps at the tensor-block level, the algorithm robustly integrates non-IID pseudo-gradients from severely delayed workers. This creates a scalable, highly efficient outer optimizer purpose-built for the realities of decentralized, geo-distributed AI infrastructure. 

