 python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts/train_dataset_met_coverage.jsonl \
    --out  output/data_generation_verithoughts/train_dataset_met_coverage_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts/pass3_removed_duplicates_verithoughts_rtl.txt

 python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts/train_dataset_not_met_coverage.jsonl \
    --out  output/data_generation_verithoughts/train_dataset_not_met_coverage_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts/pass3_removed_duplicates_verithoughts_rtl.txt


python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts/train_dataset_failed_cleaned.jsonl \
    --out  output/data_generation_verithoughts/train_dataset_failed_cleaned_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts/pass3_removed_duplicates_pyra_rtl.txt

