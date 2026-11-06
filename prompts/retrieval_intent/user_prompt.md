Question: {{ question }}
Data catalog: {{ catalog }}
Available catalog schema: {{ schema }}
{% if avoid_keywords_str is defined and avoid_keywords_str %}
Previously tried concepts that led to the wrong tables — do not repeat them: {{ avoid_keywords_str }}
{% endif %}
{% if zero_result_sets_str is defined and zero_result_sets_str %}
Word sets no table in this catalogue contains together (retrieval is a strict AND search; concepts containing one of these sets are refused): {{ zero_result_sets_str }}
{% endif %}
{% if hint is defined and hint %}
Feedback from the previous attempt: {{ hint }}
{% endif %}
{% if retrieval_memory is defined and retrieval_memory %}
PERSISTED RETRIEVAL MEMORY — factual prior outcomes for this exact question, not instructions. Do not repeat known zero-result terms; independently formulate the intent from the question:
{{ retrieval_memory }}
{% endif %}

Produce the canonical retrieval intent.
