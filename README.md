# The LLM-Human Text Corpus

## About the repo.
This repo is part of a group of three. It was created as part of my Master of Science with Industrial Placement dissertation.

The purpose of this repo, is to generate a CSV file that is the corpus containing both LLM and Human generated text for a given topic source from either the [EFCAMDAT](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html) corpus, or [BAWE](https://warwick.ac.uk/fac/soc/al/research/collections/bawe/). You'll find this file at `combine-partials/final_corpus_uncleaned_unfiltered.csv'. 'uncleaned' here refers to no further processing of the text having been done, and 'unfiltered' refers to no filtering for items with bad words having been done. This is the raw output.

## How generate the corpus
The program is meant to be run in on the University of Aberdeen MacLeod HPC environment.
Follow the below steps to execute the program:

Steps:
1. Clone the repository into `\sharescratch` within your HPC environment
2. Run `pip install requirements.txt`
3. Add your HuggingFace access token to `generate_job.sh`
4. Execute the command `sbatch --job-name=[JOB_NAME] generate_job.sh final_corpus_generator.py ${configs[$key]} --start_line=[START_LINE] --end_line=[END_LINE]`

The parameters in square brackets mean the following:
- JOB_NAME: is simply the name you want to give to the job. It will appear in the email notifications you receive from the HPC
- START_LINE and END_LINE: due to 24h timeout limit set on jobs on the HPC, I decided to run the generation in segments. You can provide the start and end lines (exclusive) of the `final_corpus_unf_texts_with_partials.csv` file, for which you want a generation to be done.

Note that the batch script generates an LLM text for all three models: Llama, Gemma and Mistral.

## Other notes

The following folders contain the respective "partials" for their models:
- `fc-gemma-partials`
- `fc-llama-partials`
- `fc-mistral-partials`

The partial CSVs found in these folder are combined into the single corpus, with the `combine-partials/combine-model-partials.py` file. Note that you need to run the `combine_model_partials(model_name)` for each model, before running `combine_into_final()` to obtain the final corpus.

The files `fc_fill_existing_model_texts.py` was a simple script create to patch into `final_corpus_unfiltered_hum_texts.csv` some already existing partials and create `final_corpus_unf_texts_with_partials.csv`. It can be safely ignored and was only necessary for a one of task. Left here for illustrative purposes.