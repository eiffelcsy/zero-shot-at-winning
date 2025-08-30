from langchain.prompts import PromptTemplate

QUERY_EXPANSION_PROMPT = """
You are helping to expand a compliance and regulatory query for better document retrieval.

Original query: "{query}"

Generate relevant additional terms and phrases that work as queries on their own and would help find documents related to this query. Focus on:
- Legal and regulatory synonyms
- Related compliance concepts
- Jurisdictional variations
- Technical implementation terms
- Related regulatory frameworks

Provide only the 5 most relevant additional terms as a comma-separated list, without explanations or the original query.
"""

QUERY_VARIATION_PROMPT = """
You are helping to generate {count} different variations of a compliance and regulatory query for comprehensive document retrieval.

Original query: "{query}"

Generate {count} distinct query variations that approach the topic from different angles:
- Different terminology and phrasing
- Various compliance perspectives (legal, technical, implementation)
- Different jurisdictional contexts
- Specific vs. general approaches
- Different stakeholder viewpoints

Format your response as a numbered list (1., 2., 3., etc.) with each variation on a new line.
Make each variation a complete, well-formed question or statement.
"""

def build_query_expansion_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["query"],
        template=QUERY_EXPANSION_PROMPT
    )

def build_query_variation_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["query", "count"],
        template=QUERY_VARIATION_PROMPT
    )
