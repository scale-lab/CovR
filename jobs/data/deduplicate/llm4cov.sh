

 python core/data/deduplicate.py   \
    --data output/data_generation_llm4cov/train_dataset_met_coverage_cleaned.jsonl \
    --out  output/data_generation_llm4cov/train_dataset_met_coverage_cleaned_ded.jsonl \
    --pass3-duplicates-out output/data_generation_llm4cov/pass3_removed_duplicates_pyra_rtl.txt


 python core/data/deduplicate.py   \
    --data output/data_generation_llm4cov/train_dataset_not_met_coverage_cleaned.jsonl \
    --out  output/data_generation_llm4cov/train_dataset_not_met_coverage_cleaned_ded.jsonl \
    --pass3-duplicates-out output/data_generation_llm4cov/pass3_removed_duplicates_pyra_rtl.txt
