def fuzzy_matching_strategy(enriched_keywords: list[str], inverted_index: dict, all_available_files: list):
    table_scores = {f: 0.0 for f in all_available_files}
    # Total number of documents (for IDF calculation)
    N = len(all_available_files) 
    for kw in enriched_keywords:
        # Track the best score for the current keyword per file to avoid "double dipping"
        # (e.g., if a file has both "residenti" and "residente", we only count the highest match once)
        kw_scores_per_file = {}
        for index_kw, files in inverted_index.items():
            score = fuzz.ratio(kw, index_kw)
            # A higher threshold (85) prevents weak/false positive matches (like 'santo' matching 'stato')
            if score >= 85:
                # Calculate Inverse Document Frequency (IDF) to give higher weight to rare keywords
                # Common words across many tables will have a lower impact.
                idf = math.log((N + 1) / (len(files) + 1)) + 1
                weighted_score = (score / 100.0) * idf
                for f in files:
                    # Keep the maximum score obtained by this specific user keyword for the file
                    kw_scores_per_file[f] = max(kw_scores_per_file.get(f, 0.0), weighted_score)