# Project Documentation: Voice Hospital Finder Bot

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Architecture Overview](#2-architecture-overview)
3.  [Key Components and Modules](#3-key-components-and-modules)
    *   [3.1 `app.py` - Application Entry Point](#31-apppy---application-entry-point)
    *   [3.2 `db/` - Database and Data Management](#32-db---database-and-data-management)
        *   [`db.py`](#dbpy)
        *   [`models.py`](#modelspy)
        *   [`modules/` - Helper Modules](#modules---helper-modules)
    *   [3.3 `graphs/` - Conversational Graph Logic](#33-graphs---conversational-graph-logic)
        *   [`graph_tools.py`](#graph_toolspy)
        *   [`hospital_graph.py`](#hospital_graphpy)
    *   [3.4 `tools/` - Utility Tools](#34-tools---utility-tools)
        *   [`rag_retrieve.py`](#rag_retrievepy)
        *   [`transcribe.py`](#transcribepy)
        *   [`recognize.py`](#recognizepy)
        *   [`text_to_speech.py`](#text_to_speechpy)
        *   [`hospital_lookup.py`](#hospital_lookuppy)
    *   [3.5 `settings/` - Configuration](#35-settings---configuration)
        *   [`config.py`](#configpy)
        *   [`prompts.py`](#promptspy)
        *   [`client.py`](#clientpy)
4.  [Setup and Installation](#4-setup-and-installation)

## 1. Introduction

The Voice Hospital Finder Bot is an intelligent conversational AI designed to assist users in finding hospitals based on their specific needs. It integrates advanced natural language processing (NLP), speech processing, and Retrieval Augmented Generation (RAG) techniques to offer a seamless voice-activated experience. The bot can understand natural language queries, search a comprehensive hospital database, and provide grounded responses, optionally enhanced by fine-tuned language models.

## 2. Architecture Overview

The system follows a modular architecture, broadly divided into:

*   **Speech Processing Layer**: Handles transcription of user's voice input and synthesis of the bot's responses.
*   **Natural Language Understanding (NLU) Layer**: Extracts intent and entities from transcribed text.
*   **Data Management Layer**: Manages hospital and insurance data, including a relational database and a FAISS vector store.
*   **Retrieval Augmented Generation (RAG) Layer**: Combines semantic search with LLM grounding to generate accurate and context-aware hospital recommendations.
*   **Conversational Orchestration Layer**: Manages the multi-turn dialogue flow using LangGraph.
*   **Configuration & Utilities**: Centralized settings and common utility functions.

## 3. Key Components and Modules

### 3.1 `app.py` - Application Entry Point

This is the main script that orchestrates the entire application. It performs the following key functions:

*   **Initialization Animation**: Provides a visual cue during startup.
*   **RAG Retriever Setup**: Initializes the `HospitalRAGRetriever` responsible for hospital data retrieval and LLM grounding. This instance is then made globally available to other components.
*   **State Management**: Initializes `HospitalFinderState` (a Pydantic model) to maintain the current state of the conversation, including user inputs, bot responses, and extracted information.
*   **Conversational Flow**: Invokes the `hospital_finder_graph` (defined in `graphs/hospital_graph.py`) to manage the conversational turns and overall logic.

### 3.2 `db/` - Database and Data Management

This directory encapsulates all functionalities related to data storage, generation, and persistence.

*   #### `db.py`
    The core database management script.
    *   **Database Setup**: Uses Pony ORM to bind to a SQLite database (`hospitals.sqlite`).
    *   **Model Definitions**: Defines `Hospital`, `InsurancePlan`, and `HospitalInsurancePlan` (many-to-many relationship) entities.
    *   **Data Population**: Automatically generates synthetic hospital and insurance plan data and populates the database if it's empty, ensuring a ready-to-use dataset.
    *   **FAISS Vector DB Creation**: If not already present, it creates a FAISS vector database from the hospital records, embedding their details for semantic search.
    *   **Fine-tuning Data Generation**: Generates dialogue-based fine-tuning data for the LLM using existing hospital and insurance information.
    *   **LLM Fine-tuning Orchestration**: Initiates the QLoRA fine-tuning process for the LLM using the generated data, if a fine-tuned model doesn't already exist.
*   #### `models.py`
    Contains Pydantic models used throughout the application for structured data validation and serialization.
    *   `LLMResponseModel`: Defines the expected structure for parsing LLM outputs, including intent, location, hospital types, etc.
    *   `RAGGroundedResponseModel`: A model to structure the response from the RAG grounding process, containing selected hospital IDs and a dialogue string.
    *   `HospitalFinderState`: Manages the state of a single conversational turn, tracking input audio paths, transcriptions, recognitions, hospitals found, turn count, and final responses.
    *   `TTSResponseModel`: Defines the structure for Text-to-Speech responses.
*   #### `modules/` - Helper Modules
    *   `fine_tuner.py`: Implements the core logic for QLoRA fine-tuning of a causal language model (e.g., Open Llama 3B V2).
        *   **Dataset Preparation**: Loads JSON data, formats it into instruction-response pairs, and tokenizes it.
        *   **Model Loading**: Loads a base LLM (e.g., `openlm-research/open_llama_3b_v2`) in 4-bit quantization using `BitsAndBytesConfig` (QLoRA).
        *   **PEFT Integration**: Applies LoRA adapters to the base model using `peft.get_peft_model`.
        *   **Training**: Configures `transformers.Trainer` with `TrainingArguments` to fine-tune the model, saving checkpoints and the final model.
    *   `fine_tune_data_generator.py`, `hospital_generator.py`, `insurance_generator.py`, `vector_db_generator.py`: These modules (inferred) are responsible for generating synthetic data, including hospital records, insurance plans, and the structured data used for fine-tuning and vector database creation.

### 3.3 `graphs/` - Conversational Graph Logic

This directory contains the LangGraph implementation that defines the conversational flow.

*   #### `graph_tools.py`
    Defines Langchain-compatible `tool`s that integrate various functionalities into the LangGraph conversational agent.
    *   `transcribe_audio_tool`: Asynchronously transcribes audio files to text using `tools.transcribe.transcribe_wrapper`.
    *   `recognize_query_tool`: Extracts structured information (intent, entities) from text using `tools.recognize.recognize_wrapper`.
    *   `text_to_speech_tool`: Converts text into speech audio using `tools.text_to_speech.text_to_speech_wrapper`.
    *   `hospital_lookup_tool`: Performs a direct lookup of hospitals based on location, type, and insurance using `tools.hospital_lookup.hospital_lookup_wrapper`.
    *   `hospital_lookup_rag_tool`: Executes a RAG-based hospital search, combining FAISS retrieval with LLM grounding using `tools.rag_retrieve.rag_search_wrapper`. It relies on a globally set `HospitalRAGRetriever` instance.
*   `hospital_graph.py`: (Inferred) This module is expected to define the nodes and edges of the LangGraph state machine, orchestrating the sequence of tool calls and state transitions based on user input and system responses.

### 3.4 `tools/` - Utility Tools

This directory contains standalone utility functions wrapped as tools.

*   #### `rag_retrieve.py`
    Implements the `HospitalRAGRetriever` class, a crucial component for the RAG functionality.
    *   **Initialization**: Loads the FAISS vector database and, if `GROUND_WITH_FINE_TUNE` is enabled, loads the fine-tuned LLM (both model and tokenizer).
    *   **Vector DB Loading**: Uses `FAISS.load_local` to load the pre-built hospital vector database.
    *   **Fine-tuned Model Loading**: Efficiently loads a PEFT-wrapped fine-tuned model (e.g., from QLoRA training) or falls back to a standard model.
    *   **`retrieve` method**: Performs a similarity search on the FAISS vector database based on a user query, filters results by distance, and sorts them by intent (nearest/best).
    *   **`ground_with_qlora` method**: Generates responses using the fine-tuned LLM, providing more specific and relevant dialogue for certain intents (e.g., `find_by_hospital`).
    *   **`ground_results` method**: The main grounding function. It dynamically decides whether to use the fine-tuned QLoRA model or a standard LLM to generate a conversational response based on retrieved hospitals and user intent. It structures the response using `RAGGroundedResponseModel`.
    *   **`rag_search_wrapper`**: An asynchronous wrapper function that orchestrates the retrieval and grounding process, returning selected hospitals and a dialogue string.
*   `transcribe.py`: (Inferred) Provides functionality for converting spoken audio into text, likely using services like OpenAI Whisper.
*   `recognize.py`: (Inferred) Implements natural language understanding capabilities to extract structured data (e.g., hospital names, locations, insurances) from free-form text. It might use spaCy for NER and potentially LLMs for more complex recognition.
*   `text_to_speech.py`: (Inferred) Handles the conversion of text into natural-sounding speech, likely using a Text-to-Speech API (e.g., OpenAI TTS).
*   `hospital_lookup.py`: (Inferred) Contains the direct logic for querying the SQLite database for hospital information, potentially used for simpler lookup scenarios or as a fallback.

### 3.5 `settings/` - Configuration

*   #### `config.py`
    A central configuration file for managing various application settings.
    *   **App Mode**: Defines the application's operating mode (`voicebot` or `chat`).
    *   **Logging**: Configures logging levels and behavior.
    *   **NLP Model**: Loads the spaCy `en_core_web_sm` model for Named Entity Recognition (NER).
    *   **Data Generation Constants**: Provides city coordinates, lists of hospital types, and insurance providers for synthetic data generation.
    *   **Tool-specific Configurations**: Contains parameters for `Transcriber`, `Recognizer`, `Clarifier`, `Text-to-Dialogue`, and `Hospital Finder` tools, including model names, temperatures, and thresholds.
    *   **RAG & Fine-tuning Paths**: Specifies file paths for the hospital database, FAISS vector database, fine-tuning data, and the fine-tuned model output directory.
*   `prompts.py`: (Inferred) Likely stores various prompts used for interacting with Large Language Models (LLMs), such as system prompts and user message templates for RAG grounding and recognition.
*   `client.py`: (Inferred) Expected to handle the initialization and configuration of external API clients, particularly for LLMs (e.g., OpenAI client, Gemini client).

## 4. Setup and Installation

Refer to the `README.md` file for detailed instructions on setting up the environment, installing dependencies, and running the application.
