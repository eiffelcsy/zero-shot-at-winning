from langchain.prompts import PromptTemplate

QUERY_EXPANSION_PROMPT = """
You are helping to expand a compliance and regulatory query for better document retrieval.

Original query: "{query}"

Generate 5 additional specific queries that would help find related compliance documents. Each query should:
- Maintain specific legal references and bill numbers when present
- Include technical implementation terms from the original
- Add related specific regulatory concepts
- Be concrete enough for precise document matching

Focus on creating variations that a compliance professional would actually search for, not generic categories.

Provide only the 5 most relevant additional queries as a comma-separated list, without explanations or the original query.
Do not generate more than 5 queries, do not use commas in the queries.

Output format: query1, query2, query3, query4, query5
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
