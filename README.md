# 📘 ROADSIGHT-RAG-KG

**End-to-End AI Decision Engine for Road Safety Interventions**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture (Deep Dive)](#-system-architecture-deep-dive)
4. [The Hybrid Engine: RAG + KG](#-the-hybrid-engine-rag--kg)
5. [Prerequisites](#-prerequisites)
6. [Installation & Setup](#-installation--setup)
7. [Configuration](#-configuration)
8. [Running the Application](#-running-the-application)
9. [API Documentation](#-api-documentation)
10. [Artifact Generation (PDF & GIS)](#-artifact-generation-pdf--gis)
11. [Testing & Quality Assurance](#-testing--quality-assurance)
12. [Project Structure](#-project-structure)

---

## 🚀 Overview

**ROADSIGHT-RAG-KG** is a specialized decision-support system designed to bridge the gap between unstructured road safety data (citizen complaints, crash narratives, police reports) and formal civil engineering interventions.

In traditional workflows, engineers manually review reports to determine if a location needs a speed bump, a roundabout, or better signage. This process is slow and subjective. **ROADSIGHT-RAG-KG** automates this by:

1.  **Ingesting** raw text and geolocation data.
2.  **Retrieving** similar past cases using Vector Search (RAG).
3.  **Validating** solutions against a Graph of Engineering Guidelines (KG).
4.  **Producing** audit-ready artifacts (PDF Cards, Map Mockups) for immediate stakeholder approval.

---

## ✨ Key Features

* **Hybrid Reasoning Engine:** Unlike standard LLM chatbots, this system will never recommend an intervention that violates the Knowledge Graph's engineering constraints (e.g., prohibiting speed humps on emergency response routes).
* **Feasibility Scoring:** Every recommendation is assigned a `0.0 - 1.0` score based on:
    * *Relevance:* Semantic match to the problem.
    * *Evidence:* Number of supporting academic/legal citations.
    * *Constraints:* Physical space and cost requirements.
* **Privacy First:** Built-in `presidio` integration automatically redacts names, license plates, and phone numbers before data hits the LLM.
* **Auditability:** The system uses a "Chain of Evidence" approach. Every output includes traceability back to the specific guideline ID (e.g., `IRC-99-2018`).
* **Multi-Modal Output:** Delivers JSON for APIs, PDF for bureaucrats, and PNG maps for visualization.

---

## 🏗 System Architecture (Deep Dive)

The architecture is designed as a unidirectional pipeline to ensure data consistency and error handling.
<img width="1453" height="711" alt="image" src="https://github.com/user-attachments/assets/46a9ee00-98d5-403c-9c70-b8038034ce6c" />
```mermaid
graph TD
    subgraph Ingestion
    A[Raw Input] -->|Sanitization| B[Normalizer]
    B -->|Keywords & Vector| C[Query Object]
    end

    subgraph Reasoning Core
    C -->|Dense Retrieval| D[FAISS Vector DB]
    C -->|Entity Linking| E[Knowledge Graph Service]
    D -->|Top-K Candidates| F[Scoring Engine]
    E -->|Evidence Paths| F
    F -->|Ranked Recs| G[Context Window]
    end

    subgraph Synthesis
    G -->|Prompt Engineering| H["LLM (Grok/OpenAI)"]
    H -->|JSON Mode| I[Structured Output]
    I -->|Schema Check| J[Validator]
    end

    subgraph Artifacts
    J -->|HTML + Jinja2| K[PDF Generator]
    J -->|Tiles + Markers| L[GIS Generator]
    K & L --> M[Final Package]
    end
