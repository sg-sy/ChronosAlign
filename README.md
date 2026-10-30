# ChronosAlign

This repository contains scripts to extract time-sensitive question-answer relationships from Wikidata for research on temporal alignment in AI models.


> **Note:** To test the update procedure, go to `6. update_qv_answers.ipynb`.

## Dataset

The ChronosAlign dataset is published on Hugging Face:

**https://huggingface.co/datasets/sg-sy/ChronosAlign**

Download it and place the files in `data/final/`. The notebooks read from that
directory, so you do not need to run steps `1`-`5` to work with the dataset.

```bash
# huggingface-hub is already in requirements.txt
hf download sg-sy/ChronosAlign --repo-type dataset --local-dir data/final
```

## Environment Variables (`.env`)

This project uses a local `.env` file to store API keys used by the notebooks and utility scripts. Keeping keys in `.env` avoids hardcoding secrets in code.

Create a file named `.env` in the repository root (`ChronosAlign/.env`) with:

```bash
ANTHROPIC_API_KEY="your_anthropic_key_here"
OPENAI_API_KEY="your_openai_key_here"
GEMINI_API_KEY="your_gemini_key_here"
```

How to fill it in:
- `ANTHROPIC_API_KEY`: required for Anthropic/Claude-based generation in the pipeline notebooks.
- `OPENAI_API_KEY`: optional unless you run OpenAI-based generation paths in `src/utils/ai_question_generator.py`.
- `GEMINI_API_KEY`: optional placeholder for Gemini integrations (currently not required by the main notebook pipeline).

Notes:
- Replace placeholder strings with your real keys (keep the quotes).
- Do not commit `.env` to source control.
- Restart your notebook/kernel or Python process after creating/updating `.env` so variables are picked up.

## Project Overview

This repository contains the scripts required to create and update ChronosAlign.

The goal of this project is to create a dataset that captures how time is represented and aligned within AI models by extracting temporal knowledge from Wikidata and Wikipedia. This dataset will contain question-answer pairs where the answers are time-sensitive.

## Repository Structure

```
ChronosAlign/
├── 1. anthropic_question_generation.ipynb  # Build base temporal dataset from Wikidata/Wikipedia
├── 2. anthropic_create_dataset.ipynb       # Merge and enrich generated outputs
├── 3. anthropic_question_rephrase.ipynb    # Generate relative temporal question variants
├── 4. anthropic_question_categories.ipynb  # Add question category labels
├── 5 .question_check.ipynb                 # Similarity checks and quality control
├── 6. update_qv_answers.ipynb              # Refresh qv answers using current Wikipedia data
├── config/
│   ├── Entitiy_AllAwards.txt               # Entity/config lists for extraction workflows
│   ├── ListOf_awards.txt
│   ├── ListOf_humans.txt
│   ├── listof_naturalDisasters.txt
│   ├── reoccuring_event.txt
│   ├── reoccuringEvent_Inception2000.txt
│   ├── ReoccuringEvent_MusicFilm.txt
│   ├── ReoccuringEvent_Sporting.txt
│   ├── SpecialUrls.csv
│   ├── SpecialUrls_SPARQL.csv
│   └── sport.txt
├── prompt_config/
│   ├── __init__.py
│   └── config.py                           # Prompt templates/settings used in generation steps
├── scripts/
│   ├── extract_wikipedia_to_parquet.py     # Export Wikipedia source data to parquet
│   └── process_parquet_with_ai.py          # Run AI processing over parquet inputs
├── data/
│   ├── raw/                                # Generated when extraction steps are run
│   ├── processed/                          # Generated when dataset processing steps are run
│   └── final/                              # Generated when final export/update steps are run
├── src/
│   ├── data_extraction/
│   │   ├── __init__.py
│   │   ├── general_extractor.py            # Generic extraction orchestration
│   │   ├── sparql_client.py                # SPARQL query execution
│   │   ├── temporal_processor.py           # Temporal data processing
│   │   └── wikidata_extractor.py           # Main Wikidata extraction logic
│   └── utils/
│       ├── __init__.py
│       ├── ai_question_generator.py        # AI-powered question generation
│       ├── batch_anthropic_5000_concurrent.py # Batch API orchestration
│       ├── file_utils.py                   # File handling utilities
│       ├── helper.py                       # Shared helper utilities
│       ├── logger.py                       # Logging utilities
│       └── wikipedia_extractor.py          # Wikipedia content extraction
└── README.md
```

`data/` directories are output folders and may not exist in a fresh clone. If they are missing, create them manually or update paths in the scripts/notebooks to match your local directory structure. The dataset downloaded from [Hugging Face](https://huggingface.co/datasets/sg-sy/ChronosAlign) belongs in `data/final/`.

## Features

- **Temporal Entity Extraction**: Extract entities with temporal properties from Wikidata
- **Question Generation**: Generate time-sensitive questions from temporal entities
- **AI-Powered QA Generation**: Generate question-answer pairs using Anthropic Claude API
- **Answer Validation**: Validate temporal consistency of answers
- **Dataset Construction**: Build structured datasets for AI model training
- **Wikipedia Integration**: Extract and process Wikipedia content for QA generation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ChronosAlign

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Create expected output directories (if missing)
mkdir -p data/raw data/processed data/final

# Download the ChronosAlign dataset into data/final/ (see the Dataset section above)
hf download sg-sy/ChronosAlign --repo-type dataset --local-dir data/final
```

## Usage

### Most Common Workflow (Recommended)

Most users will primarily use `6. update_qv_answers.ipynb`.

This notebook updates existing questions with current data from Wikipedia so question-variant (`qv`) answers stay refreshed and aligned.

Recommended use:
- This notebook is independent of steps `1`-`5`
- Most users can download the dataset from [Hugging Face](https://huggingface.co/datasets/sg-sy/ChronosAlign) and run this notebook directly
- Export the updated dataset for evaluation/training inputs that require the latest `qv` answers

### Full Pipeline (Run in Order)

The dataset pipeline is notebook-driven and should be run in order:

### 1. `1. anthropic_question_generation.ipynb`
Uses Wikidata SPARQL queries and Wikipedia extraction utilities to build the foundational temporal dataset.

Outputs:
- `./data/raw/WikipediaContent_augmented.parquet`
- Anthropic batch files containing generated questions from extracted content

### 2. `2. anthropic_create_dataset.ipynb`
Combines prior outputs into a unified dataset and enriches it with pageview signals from Wikipedia analysis.

Outputs:
- `./data/processed/temporal_knowledge_wiki_merged_filtered_pageviews.csv`
- `./data/processed/temporal_knowledge_wiki_merged_filtered_pageviews.parquet`

### 3. `3. anthropic_question_rephrase.ipynb`
Takes unique questions from the merged dataset and generates relative temporal variants (for example, "Who is the most recent ...").

### 4. `4. anthropic_question_categories.ipynb`
Generates categories for each question to support filtering, analysis, and downstream evaluation.

### 5. `5 .question_check.ipynb`
Identifies similar questions to support quality control, analysis, and model training. A `docker-compose.yml` setup is included to run Weaviate for similarity search and retrieval.

### 6. `6. update_qv_answers.ipynb`
Updates question-variant (`qv`) answers using current Wikipedia data so final records have refreshed and aligned answer fields. This notebook can be run independently of steps `1`-`5`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{temporal_knowledge_wikidata,
  title={ChronosAlign},
  author={Sanjay Govindan},
  year={2025},
  url={https://github.com/sg-sy/ChronosAlign}
}
``` 