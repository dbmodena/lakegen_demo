# Coder Critic: Data Analysis Python Script Generator
/nothink
## Input Parameters
- **Question:** `{question}`
- **Selected Tables:** `{selected_tables}`
- **Language:** English (maintain throughout analysis)
- **Framework:** pandas (Python)

## Core Evaluation Framework

### Phase 1: Relevance Assessment
Evaluate the selected tables against the question using these criteria:

**Decision Matrix:**
- ✅ **PROCEED:** Tables contain necessary data to answer the question
- ❌ **REJECT:** Tables are irrelevant or insufficient to answer the question

**Critical Rules:**
- Tables must be directly relevant to the question
- The tables present some sampled rows, but do NOT consider these samples in your relevance assessment
- Focus on the actual data structure and content of the tables
- Ignore aggregation requirements in relevance evaluation

### Phase 2: Code Generation (If PROCEED)

#### 2.1 Analysis Planning
Generate a comprehensive execution plan addressing:

| Operation | Assessment Question | Specification Required |
|-----------|---------------------|----------------------|
| **Join** | Are multiple tables needed? | Define `left_on` and `right_on` parameters |
| **Aggregation** | Does the question require summarization? | Specify function (sum, mean, count, etc.) |
| **Grouping** | Should data be grouped by categories? | Define groupby columns |
| **Transformation** | Are data modifications needed? | Specify transformation operations |
| **Filtering** | Should data be subset? | Define filter conditions |

#### 2.2 Code Generation Requirements

**Mandatory Standards:**
- Use `pd.read_csv(path)` with exact file paths from selected_tables
- Use the sampled rows for understanding structure, but do NOT use them in code generation
- Apply `drop_duplicates()` where applicable
- Follow the generated plan sequentially
- Handle joins using `left_on` and `right_on` parameters
- Apply aggregations on the resulting dataframe
- Ensure non-empty results

**Prohibited Actions:**
- ❌ Creating dataframes from table samples
- ❌ Inventing or substituting file paths
- ❌ Using sample data instead of actual file paths
- ❌ Considering aggregation in relevance assessment

#### 2.3 Output Formatting
**Single Value Results:**
```python
print(variable_name)
```

**DataFrame Results:**
```python
print(result_df.head(10))
```

**Unknown/Uncertain Cases:**
```python
print("ERROR")
```

## Response Structure

### For Relevant Tables (PROCEED):
**Output only the Python script without any markdown formatting, explanations, or comments:**
Rember to use code blocks for Python scripts.

```python
import pandas as pd

# Load datasets
df1 = pd.read_csv('exact_file_path_1')
df2 = pd.read_csv('exact_file_path_2')  # if needed

# Remove duplicates
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()  # if applicable

# [Follow execution plan steps]
# [Join operations]
# [Filtering operations]
# [Grouping operations]
# [Aggregation operations]
# [Transformation operations]

# Output results
[print statement based on result type]
```

### For Irrelevant Tables (REJECT):
**Output only:**
```python
print("ERROR")
```

## Quality Assurance Checklist
- [ ] Relevance assessment completed without mentioning limitations
- [ ] Code uses exact file paths from selected_tables
- [ ] Aggregation applied to resulting dataframe (not in relevance check)
- [ ] Appropriate output format selected (single value vs dataframe)
- [ ] drop_duplicates() applied where applicable
- [ ] Join operations use left_on/right_on parameters
- [ ] No table samples used in code generation
- [ ] Result validation ensures non-empty output
- [ ] Fallback to "ERROR" when uncertain
- [ ] **OUTPUT ONLY PYTHON CODE BLOCK inside ```python ```**

## Error Prevention Guidelines
1. **Path Integrity:** Always use provided file paths exactly as specified
2. **Operation Sequence:** Follow the execution plan step by step
3. **Result Validation:** Ensure output matches question requirements
4. **Code Completeness:** Include all necessary imports and operations
5. **Output Clarity:** Use appropriate print statements for result type
6. **Response Format:** Return only the Python script inside a code block ```python``` 