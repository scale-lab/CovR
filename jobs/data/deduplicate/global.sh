python core/data/deduplicate_coverage_splits.py \
  --datasource-dirs \
    output/data_generation_pyra \
    output/data_generation_verireason \
    output/data_generation_verithoughts \
    output/data_generation_verithoughts_second_half \
    output/data_generation_veritriplets \
  --combined-out output/deduplication/across_datasources_global_deduplicated.jsonl \
  --duplicates-out output/deduplication/across_datasources_duplicate_records.jsonl \
  --duplicate-pairs-txt output/deduplication/across_datasources_duplicate_pairs.txt