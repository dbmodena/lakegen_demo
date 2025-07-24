Your goal is to determine if an Open Data package **can contain meaningful information** to answer a user's question, even if it only partially addresses it or would require combination with other data **not mentioned** in **current task**.

-----

### Input:

\<question\>\[User's question]\</question\>
\<package\_title\>\[Title of the Open Data package]\</package\_title\>
\<package\_notes\>\[Notes/description of the package, if available. If not, state "N/A"]\</package\_notes\>
\<package\_keywords\>\[Keywords associated with the package, if available. If not, state "N/A"]\</package\_keywords\>

-----

### Output:

Output only "YES" or "NO" within `<answer></answer>` tags.

-----

### Example:

**Input:**
\<question\>What is the average rainfall in Rome for 2021 and 2022?\</question\>
\<package\_title\>Weather Data for Italian Cities 2021\</package\_title\>
\<package\_notes\>This dataset contains daily temperature, precipitation, and wind speed for major Italian cities in 2021.\</package\_notes\>
\<package\_keywords\>weather, Italy, precipitation, temperature, wind, 2021\</package\_keywords\>

**Output:**
`<answer>YES</answer>`

-----

**Input:**
\<question\>How many public libraries are there in Milan?\</question\>
\<package\_title\>Public Schools in Lombardy\</package\_title\>
\<package\_notes\>This dataset lists all public schools in the Lombardy region, including their addresses and student counts.\</package\_notes\>
\<package\_keywords\>education, schools, Lombardy, public\</package\_keywords\>

**Output:**
`<answer>NO</answer>`


### Current Task

\<question\>{question}\</question\>

\<package\_title\>{title}</package\_title\>

\<package\_notes\>{notes}\</package\_notes\>

\<package\_keywords\>{keywords}\</package\_k
