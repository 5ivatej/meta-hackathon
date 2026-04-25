# OpenEnv Hackathon 2026 – Rules, Themes, and Evaluation

## Overview

This hackathon focuses on building **training environments for LLM agents** that enable measurable improvement in complex capabilities such as reasoning, planning, interaction, and self-improvement.

The goal is not just to build an environment, but to:
- Train an agent inside it
- Demonstrate measurable improvement
- Clearly communicate the problem, approach, and results

---

## Core Themes

### Theme #1: Multi-Agent Interactions

**Focus:**
- Cooperation, competition, negotiation, coalition formation
- Modeling beliefs and incentives of other agents (Theory of Mind)

**Expected Outcome:**
An environment for training LLMs on **multi-agent task handling**

**Example Environments:**
- Market simulations
- Compute allocation negotiation systems
- Collaborative puzzle worlds
- Mixed cooperative/competitive games

---

### Theme #2: Long-Horizon Planning & Instruction Following

**Focus:**
- Multi-step reasoning
- Sparse/delayed rewards
- Tracking state over long trajectories
- Recovering from early mistakes

**Expected Outcome:**
Environment enabling LLMs to improve on **long-running, complex tasks beyond context limits**

**Example Environments:**
- Research planning simulators
- Codebase refactoring systems
- Logistics/resource optimization worlds
- Multi-turn workflows (e.g., 300-step instruction chains)

---

### Theme #3: World Modeling

#### #3.1 Professional Tasks

**Focus:**
- Real-world tool/API interactions
- Persistent state and causal reasoning
- Multi-step workflows

**Expected Outcome:**
Environment that improves LLM ability to operate in **dynamic, partially observable systems**

**Example Environments:**
- Browser/API ecosystems
- Enterprise workflows
- Scientific loops (papers → code → experiments)
- Economic simulations with feedback

---

#### #3.2 Personalized Tasks

**Focus:**
- Personal decision-making
- Conflict resolution
- Delegation and prioritization

**Expected Outcome:**
Environment for **realistic personal assistant behavior**

**Example Environments:**
- Meeting planners
- Dinner/work conflict managers
- Email/message response systems
- Shopping/task delegation systems

---

### Theme #4: Self-Improvement

**Focus:**
- Self-play
- Curriculum generation
- Recursive capability improvement

**Expected Outcome:**
Environment where agents **improve themselves over time**

**Example Environments:**
- Self-play negotiation arenas
- Auto-generated math/proof tasks
- Evolving coding challenges
- Adaptive RL curricula

---

### Theme #5: Wild Card

**Focus:**
- Any novel idea outside defined themes

**Requirement:**
Must still contribute meaningfully to **LLM training and evaluation**

---

## Problem Statement Requirements

Your submission must clearly define:

- Problem Statement
- Environment Design
- Agent Capabilities
- Tasks
- Reward Model / Evaluation Logic
- Post-Training or Self-Improvement Strategy

---

## Minimum Submission Requirements

These are **mandatory**:

- Use **OpenEnv (latest release)**
- Provide **training script** (Unsloth or HuggingFace TRL)
- Preferably runnable via **Colab**
- Show **actual training evidence**
  - Loss curves
  - Reward curves
- Publish environment on **Hugging Face Spaces**
- Provide a **README** including:
  - Problem explanation
  - Environment details
  - Results
  - Links to:
    - Blog / video (<2 min)
    - Plots / demos

---

## Judging Criteria

### 1. Environment Innovation (40%)

- Novelty and creativity
- Complexity and challenge
- Does it test meaningful agent behavior?

---

### 2. Storytelling & Presentation (30%)

- Clarity of explanation
- Demo quality
- Accessibility to non-technical audience

---

### 3. Showing Improvement in Rewards (20%)

- Evidence of learning
- Before vs after comparisons
- Reward curves / metrics

---

### 4. Reward & Training Pipeline (10%)

- Quality of reward design
- Effectiveness of training pipeline
- Alignment between reward and behavior

---

## What Makes a Strong Submission

### 1. Ambitious Problem Selection

Ask:
- Does this teach something LLMs currently struggle with?
- Is this underexplored?
- Could this become a research paper?

---

### 2. Strong Reward Design

A good reward function:
- Provides dense or meaningful feedback
- Is hard to exploit
- Reflects real task success
- Uses composable rubric-based scoring

---

### 3. Real Training Evidence

You must:
- Train against your environment (not static data)
- Run for sufficient steps
- Show:
  - Baseline vs trained agent
  - Quantitative + qualitative improvements

---

### 4. Clear Visualizations

- Label axes clearly
- Include units
- Use readable formats (.png/.jpg)
- Compare runs on same plot
- Embed in README

---

### 5. Strong Storytelling

Your submission should answer:

- **Problem:** What gap are you solving?
- **Environment:** What does the agent see/do?
- **Results:** What improved after training?
- **Impact:** Why does this matter?

---

### 6. Clean Engineering (Baseline Expectation)

- Use OpenEnv base classes properly
- Follow Gym-style API:
  - `reset()`
  - `step()`
  - `state`
- Maintain client/server separation
- Include valid `openenv.yaml`
- Avoid reserved tool names

---

## Final Guidance

- Prefer **ambitious + messy** over polished but trivial
- Focus on **training impact**, not just environment design
- Ensure **reproducibility**
- Keep README **clear and fast to read (3–5 mins)**

---

## TL;DR

Build:
- A meaningful environment  
- That trains an LLM  
- Shows measurable improvement  
- And tells a clear story
