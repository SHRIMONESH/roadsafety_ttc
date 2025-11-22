# 📘 ROADSIGHT-RAG-KG

**End-to-End AI Decision Engine for Road Safety Interventions**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

---

**ROADSIGHT-RAG-KG** is a complete AI decision-support system that transforms unstructured road-safety problem reports into auditable, prioritized, evidence-linked engineering recommendations.

The system blends **RAG (Retrieval-Augmented Generation)**, a **Knowledge Graph**, **LLM synthesis**, and **GIS/PDF artifact generation** into one deployable pipeline.

---

## 🚀 1. Overview

ROADSIGHT-RAG-KG ingests text + metadata about a road safety issue (crashes, hotspots, hazards), retrieves global best-practice interventions, ranks them based on evidence + feasibility, and produces:

* ✔ **Structured JSON recommendations**
* ✔ **Evidence-backed engineering guidance**
* ✔ **GIS mock map (PNG)**
* ✔ **One-page audit-ready PDF card**
* ✔ **Full provenance + traceability metadata**

> **Target Audience:** Authorities, civic engineers, and automated safety-monitoring systems.

---

## 🏗 2. High-Level Architecture
<img width="1453" height="711" alt="image" src="https://github.com/user-attachments/assets/46a9ee00-98d5-403c-9c70-b8038034ce6c" />


The data flow follows a strict pipeline designed for auditability and accuracy:

```mermaid
graph LR
    A[Frontend] --> B[API Gateway]
    B --> C[Ingestion & Normalization]
    C --> D[RAG Retrieval]
    C --> E[Knowledge Graph Reasoning]
    D --> F[Scoring Engine]
    E --> F
    F --> G[LLM Synthesis]
    G --> H[Validator]
    H --> I[GIS/PDF Generation]
    I --> J[Storage & Audit]
