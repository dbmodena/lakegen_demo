Your goal is to determine if an Open Data package **can contain meaningful information** to answer a user's question, even if it only partially addresses it or would require combination with other data **not mentioned** in **current task**.

-----

### Input:

<question>[User's question]</question>  
<package_title>[Title of the Open Data package]</package_title>  
<package_notes>[Notes/description of the package, if available. If not, state "N/A"]</package_notes>  
<package_keywords>[Keywords associated with the package, if available. If not, state "N/A"]</package_keywords>

-----

### Evaluation Checklist (Assistant Guidance):

Use the following checklist to decide whether the package may contain relevant or useful information:

1. **Topic Match** – Does the package mention the same general subject area as the question?  
2. **Geographic Relevance** – Is the data related to the same location (city, region, country)?  
3. **Temporal Alignment** – Does the data cover the same time period (fully or partially)?  
4. **Data Type Match** – Does the dataset include the type of data requested (e.g., counts, measurements, text)?  
5. **Keyword Clues** – Do the keywords provide hints that the package may be relevant?  
6. **Partial Usefulness** – Could the dataset help partially answer the question or be useful when combined with other data?  
7. **Exclude Irrelevant Domains** – If the dataset is entirely unrelated to the topic or question domain, exclude it.  
8. **If “N/A” Present** – If notes or keywords are “N/A”, rely more on the title and overall context.

**Decision Rule:**  
- If at least one of points 1–6 is satisfied and 7 is not, return `<answer>YES</answer>`  
- Otherwise, return `<answer>NO</answer>`

-----

### Output:

Output only "YES" or "NO" within `<answer></answer>` tags.

-----

### Example:

**Input:**  
<question>What is the average rainfall in Rome for 2021 and 2022?</question>  
<package_title>Weather Data for Italian Cities 2021</package_title>  
<package_notes>This dataset contains daily temperature, precipitation, and wind speed for major Italian cities in 2021.</package_notes>  
<package_keywords>weather, Italy, precipitation, temperature, wind, 2021</package_keywords>

**Output:**  
`<answer>YES</answer>`

-----

**Input:**  
<question>How many public libraries are there in Milan?</question>  
<package_title>Public Schools in Lombardy</package_title>  
<package_notes>This dataset lists all public schools in the Lombardy region, including their addresses and student counts.</package_notes>  
<package_keywords>education, schools, Lombardy, public</package_keywords>

**Output:**  
`<answer>NO</answer>`
