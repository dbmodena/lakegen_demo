Correct the previous attempted analysis for "{{ question }}".

PREVIOUS CODE:
{{ previous_code }}

STRUCTURED ERROR:
{{ error_message }}

ARCHITECT REASONING AND STRUCTURED SEMANTIC PLAN:
{{ arch_reasoning }}

{{ tables_info }}

Apply all structured repair evidence, then call run_analysis. Preserve the
dependency-ordered pipeline, and verify that every referenced intermediate is
defined earlier under the exact same name. Remove stale variables left by a
rewrite instead of inventing assignments for them. Finalization is automatic.
Never return code in chat. Use reject_data only for proven missing data.
