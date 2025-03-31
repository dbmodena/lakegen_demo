# <img src="frontend/static/images/image.png" alt="Image Alt Text" style="width: 50px; vertical-align: middle;"> LakeGen Demo


# LakeGen
<video controls>
  <source src="frontend/static/images/demo.mp4" type="video/mp4">
</video>

## Abstract
The increasing availability of tabular data in open data lakes presents significant opportunities for data-driven analysis. 
Yet, it also introduces substantial challenges in data discovery and integration, as users must contend with diverse formats, inconsistent schemas, and incomplete metadata—issues that complicate the retrieval of relevant datasets and the extraction of meaningful insights. 
Traditional dataset retrieval methods rely on structured indexing and manual selection, which are not feasible for large, dynamic open data collections.
Similarly, existing table question-answering methods assume predefined schemas and curated tables, limiting their applicability in open data environments.

We introduce **LakeGen**, a multi-agent framework designed to automate the retrieval and reasoning over open datasets accessed via APIs. 
By dynamically identifying relevant tables, integrating heterogeneous data sources, and structuring responses, LakeGen enables users to perform complex analyses without requiring expertise in database querying.

In this demonstration, we show how LakeGen can simplify the complexities of working with open data lakes, allowing attendees to efficiently extract insights without prior knowledge of the underlying data.

## Running the UI
To start the UI, run the following command:

```sh
./start.sh
```

## Generating an API Key on Groq Cloud
To use LakeGen with Groq Cloud, you will need an API key. Follow these steps to create one:

1. Visit the Groq Cloud API key creation page: [Create an API Key](https://console.groq.com/keys)
2. Log in to your Groq Cloud account or create a new account if you don't already have one.
3. Once logged in, follow the instructions on the page to generate your API key.
4. Copy the generated API key and store it securely.

You can then use this key to interact with LakeGen and Groq Cloud APIs.

