 python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts_second_half/train_dataset_met_coverage.jsonl \
    --out  output/data_generation_verithoughts_second_half/train_dataset_met_coverage_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts_second_half/pass3_removed_duplicates_verithoughts_rtl.txt

 python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts_second_half/train_dataset_not_met_coverage.jsonl \
    --out  output/data_generation_verithoughts_second_half/train_dataset_not_met_coverage_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts_second_half/pass3_removed_duplicates_verithoughts_rtl.txt


python core/data/deduplicate.py   \
    --data output/data_generation_verithoughts_second_half/train_dataset_failed_cleaned.jsonl \
    --out  output/data_generation_verithoughts_second_half/train_dataset_failed_cleaned_deduplicated.jsonl \
    --pass3-duplicates-out output/data_generation_verithoughts_second_half/pass3_removed_duplicates_pyra_rtl.txt

python core/data/deduplicate_coverage_splits.py \
  --met output/data_generation_verithoughts_second_half/train_dataset_met_coverage.jsonl \
  --not-met output/data_generation_verithoughts_second_half/train_dataset_not_met_coverage.jsonl \
  --met-out output/data_generation_verithoughts_second_half/train_dataset_met_coverage_deduplicated.jsonl \
  --not-met-out output/data_generation_verithoughts_second_half/train_dataset_not_met_coverage_deduplicated.jsonl \
  --duplicates-out output/data_generation_verithoughts_second_half/met_not_met_duplicate_records.jsonl \
  --duplicate-pairs-txt output/data_generation_verithoughts_second_half/met_not_met_duplicate_pairs.txt