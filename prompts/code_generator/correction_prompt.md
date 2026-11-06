The previous code for "{{ question }}" failed with an error.

PREVIOUS CODE:
```python
{{previous_code}}
```

ERROR: "{{ error_message }}"

If ERROR contains STRUCTURED_REPAIR_CONTEXT, treat that JSON as authoritative.
Apply repair_hint and rename_hints exactly. A source column may have been
renamed or removed downstream even when it exists in AVAILABLE TABLES.

ARCHITECT REASONING: "{{ arch_reasoning }}"

{{ tables_info }}

FIX RULES:
- Security Error / forbidden code fragment → remove that import or operation completely; do not replace it with another system/process library.
- KeyError → use the EXACT column name from AVAILABLE TABLES (do NOT print df.columns to debug).
- Missing required columns → compare the previous name with AVAILABLE TABLES and preserve spaces, slashes, hyphens, and all other punctuation exactly.
- FileNotFoundError → use the EXACT LOAD command from AVAILABLE TABLES.
- CARTESIAN_PRODUCT_DETECTED → the join key is not unique on one or both sides. Aggregate that side with `.groupby(key).agg(...)` before merging, or find the correct unique key.
- Address every reported error explicitly before rewriting; do not reproduce the failing column literals or forbidden imports from PREVIOUS CODE.
- Rewrite the complete working script. Reply ONLY with the code block.
