#  Agent-Based Model for Population Dynamics of spider monkeys

This repository contains the source code for an agent-based model studying the effects of forest deforestation on brown-headed spider monkey populations. This repository contains the interactive version of the project produced to be presented in conferences.

---

## 🛠 Requirements

This project is written in **Julia** and uses the following packages:

- Agents
- CSV
- GLMakie
- CairoMakie
- DataFrames
- Dates
- Random
- StatsBase

Recommended Julia version: **1.10.10**

---

## ⚙️ Installation

Start Julia:

```bash
julia
```

Install the required packages:
```Julia
using Pkg
Pkg.add(["Agents", "CSV", "GLMakie", "CairoMakie", "DataFrames", "Dates", "Random", "StatsBase"])
```

---

▶️ How to Run the Model

Option 1 – Run from the terminal:

```bash
julia main/multiple_territories_interactive.jl
```

Option 2 – Run from the Julia REPL:

```Julia
cd("main")
include("multiple_territories_interactive.jl")
```
---
