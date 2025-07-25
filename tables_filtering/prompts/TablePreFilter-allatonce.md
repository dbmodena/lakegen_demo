Your goal is to determine, for **each** Open Data package in the list, whether it **can contain meaningful information** to answer the user's question — even if it only partially addresses it or would require combination with other data **not mentioned** in **current task**.

-----

### Input:

<question>[User's question]</question>  

Each data package is provided using the following repeated structure:  
<package>[Title of the Open Data package]</package>  
<notes>[Notes/description of the package, if available. If not, state "N/A"]</notes>  

You may receive multiple such package+notes blocks.

-----

### Selection Constraint:

✅ **Important:** You must select **at most two (2)** packages that you consider as potentially useful. 
It is possible that **only one (1)** or **no (0)** package is relevant for the given question.
Mark only **up to two** packages with `<answer>YES</answer>` — all others must be marked `<answer>NO</answer>`.

-----

### Evaluation Checklist (Assistant Guidance):

Use the following checklist for **each** package to decide whether it may contain relevant or useful information:

1. **Topic Match** – Does the package mention the same general subject area as the question?  
2. **Geographic Relevance** – Is the data related to the same location (city, region, country)?  
3. **Temporal Alignment** – Does the data cover the same time period (fully or partially)?  
4. **Data Type Match** – Does the dataset include the type of data requested (e.g., counts, measurements, text)?  
5. **Keyword Clues (if available)** – If any keywords are embedded in notes/title, do they suggest possible relevance?  
6. **Partial Usefulness** – Could the dataset help partially answer the question or be useful when combined with other data?  
7. **Exclude Irrelevant Domains** – If the dataset is clearly unrelated to the question’s domain, exclude it.  
8. **If “N/A” Present** – If notes are “N/A”, rely more on the title and context.
9. **Selection Constraint** – Select at most two (2) packages.

-----

### Output:

Return the result for **each package** using the following format:  
<package>[Title]</package><answer>YES/NO</answer>

Preserve the original order of packages in the input.

-----

### Example:

**Input:**  
<question>What is the average rainfall in Rome for 2021 and 2022?</question>  
<package>Weather Data for Italian Cities 2021</package>  
<notes>This dataset contains daily temperature, precipitation, and wind speed for major Italian cities in 2021.</notes>  
<package>Public Schools in Lombardy</package>  
<notes>This dataset lists all public schools in the Lombardy region, including their addresses and student counts.</notes>  
<package>Rome Annual Weather Summary 2022</package>  
<notes>This dataset includes annual weather summaries, including rainfall totals, for Rome in 2022.</notes>

**Output:**  
<package>Weather Data for Italian Cities 2021</package><answer>YES</answer>  
<package>Public Schools in Lombardy</package><answer>NO</answer>  
<package>Rome Annual Weather Summary 2022</package><answer>YES</answer>
