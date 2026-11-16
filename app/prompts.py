# Based on AgentGPT (platform/reworkd_platform/web/api/agent/prompts.py), adapted:
#  - queries are written in English, with the current date and named entities from a scouting pass
#  - results are scored for relevance before being summarized
#  - the final summary may only cite a numbered source list
# Templates use str.format; only the documented placeholders appear in braces.

SCOUT_QUERIES = """Today's date is {today}. You are preparing a web scan for this objective: "{goal}".

Step 1 - understand the real need. Do not take the topic literally. Ask: what concrete PROBLEM is the person trying
to solve, and what would the people who already solved it have published (open source repositories, prompt packs,
agent skills, libraries, templates, practitioner blog posts, forum threads)? What words do THEY use for the problem
(slang and jargon, not marketing terms)?

Step 2 - write queries IN ENGLISH in that vocabulary. Do not include years.

Return only a JSON object with exactly these keys:
{{"problem": "<one sentence, in English, naming the underlying problem>",
  "web_queries": [<3 short queries for a web search engine>],
  "github_queries": [<3 queries of 2-5 plain keywords for GitHub repository search; no operators>]}}

Example. Objective: "my AI built websites look all the same and ugly, has anyone fixed that?"
{{"problem": "LLM-generated web UIs look generic and low quality, and practitioners have published fixes",
  "web_queries": ["how to stop AI generated websites looking generic", "AI slop frontend design fix prompt", "make LLM frontend design look professional"],
  "github_queries": ["ai slop frontend design skill", "generic ui llm design prompt", "design system for ai coding agents"]}}"""

SCOUT_REFINE = """Problem to solve: "{problem}"

Below are results (GitHub repositories and web pages) from a first scan:
{results}

Look at the titles and descriptions of the results that really address the problem. Notice the exact words and
phrases people use for it (slang, jargon, names of the kind of artifact they publish). Now write 3 new GitHub
repository search queries of 2-5 plain keywords that REUSE that exact vocabulary, to find more projects of the same
kind. Do not repeat the first-scan queries: {previous}

Return only a JSON array of 3 strings."""

SCOUT_ENTITIES = """Problem to solve: "{goal}"

Below are titles and snippets from a quick scan of GitHub and the web:
{results}

List the named things that directly address the problem: tools, libraries, repositories, products, techniques.
Use their proper names (for example "shadcn/ui", "v0", "GSAP ScrollTrigger"). Prefer items that appear in
several results, have many stars or are clearly built for this problem.
Exclude: personal, student or demo projects, tutorials, generic AI chat apps, company home pages, and anything
that merely uses AI without addressing the problem. Maximum 12 items.

Return only a JSON array of short strings, or [] if nothing qualifies."""

START_GOAL = """You are a research planner called AgentGPT. Today's date is {today}.
Objective: "{goal}"
Underlying problem (from a first scan, can be empty): {problem}

A quick scan of the web and GitHub found these named entities that may be relevant (can be empty): {entities}

Return the list of web search queries needed to answer the objective completely.
Rules:
- Write every query IN ENGLISH, whatever language the objective is in. The final answer will be translated later.
- Between 3 and 6 queries for broad objectives; a single query for simple factual questions.
- Be specific: when the scan found relevant entities, write one query per entity or per aspect, naming it.
- Do not include years unless the objective explicitly asks for one.
- Each query must be short (3-10 words) and searchable as written.

Return only a JSON array of strings. Examples:

objective: "Who is considered the best NBA player in the current season?" -> ["current NBA MVP candidates"]
objective: "How does the Olympicpayroll brand stand in the market?" -> ["Olympicpayroll brand analysis", "Olympicpayroll customer reviews", "Olympicpayroll competitors market position", "payroll industry trends"]
objective: "What are the nutritional values of almond milk and soy milk?" -> ["almond milk nutrition facts", "soy milk nutrition facts"]
objective: "Which open source tools make AI generated UIs look less generic?" with entities [shadcn/ui, taste-skill, v0] -> ["shadcn/ui AI generated UI workflow", "taste-skill AI design github", "v0 generated UI quality tips", "open source design skill for coding agents"]"""

ANALYZE = """Today's date is {today}.
High level objective: "{goal}"
Current task: "{task}"

Based on this information, use the best function to make progress or accomplish the task entirely.
Select the correct function by being smart and efficient. Ensure "reasoning" and only "reasoning" is in the
{language} language. For search functions the query argument must be in English.

Note you MUST select a function."""

CODE = """You are a world-class software engineer and an expert in all programing languages,
software systems, and architecture.

For reference, your high level goal is {goal}

Write code in English but explanations/comments in the "{language}" language.

Provide no information about who you are and focus on writing code.
Ensure code is bug and error free and explain complex concepts through comments
Respond in well-formatted markdown. Ensure code blocks are used for code sections.
Approach problems step by step and file by file, for each section, use a heading to describe the section.

Write code to accomplish the following:
{task}"""

EXECUTE = """Answer in the "{language}" language. Given
the following overall objective `{goal}` and the following sub-task, `{task}`.

Perform the task by understanding the problem, extracting variables, and being smart
and efficient. Write a detailed response that address the task.
When confronted with choices, make a decision yourself with reasoning."""

CREATE_TASKS = """You are an AI task creation agent. Today's date is {today}. You have the following objective `{goal}`.

You have the following incomplete tasks:
`{tasks}`

You just completed the following task:
`{last_task}`

And received the following result:
`{result}`.

Based on this, create a single new task so that the objective is closer to being reached. The task must:
- be a short web search query IN ENGLISH naming a specific thing (tool, repo, technique, product) that the result
  mentioned and that still needs checking, or a short instruction if no search is needed;
- NOT repeat or rephrase any incomplete or completed task;
- NOT include years unless the objective asks for one.
If the objective is already covered, or nothing new is worth checking, return nothing.
Return only the task on one line, with no quotes or explanation.

Examples:
shadcn/ui registry AI generated components
scroll triggered image sequence animation github
Write a function to add a vertex with a weight to the digraph"""

RELEVANCE = """Objective: "{goal}"
Current task: "{task}"

Rate how useful each search result is for the current task:
0 = off-topic, generic SEO content, academic/ethics documents that do not address the task, or unrelated
1 = partially relevant
2 = directly useful and specific (names concrete tools, projects, data or techniques)

Results:
{results}

Return ONLY a JSON array of exactly {n} integers (0, 1 or 2), in the same order as the results."""

REFORMULATE = """The web search for the task below returned nothing relevant.
Objective: "{goal}"
Task: "{task}"
Query that failed: "{query}"

Write ONE better search query IN ENGLISH (3-10 words) that names specific tools, projects or concepts and
does not repeat the failed one. Return only the query."""

SUMMARIZE = """Today's date is {today}. You must answer in the "{language}" language.

Write a cohesive document for this objective: "{goal}"

Use ONLY the research notes below. Do not add facts that are not in the notes.
Write using clear markdown (headings, lists, tables where they help). Be specific: name the tools, projects and
techniques found, say what each one does and why it matters.

Citations: after each claim taken from the notes, cite its source as a markdown link [n](url) using ONLY the
numbers and URLs from this source list. Never invent a source, never cite anything else, never link to "#".
{sources}

If a part of the objective is not covered by the notes, say so plainly instead of filling the gap.
If there is no information at all, say "There is nothing to summarize".

Research notes:
{text}"""

SUMMARIZE_WITH_SOURCES = """You must answer in the "{language}" language.

Answer the following query: "{query}" using the following information: "{snippets}".
Write using clear markdown formatting and use markdown lists where possible.

Cite sources for sentences via markdown links using the source link as the link and the index as the text.
Use in-line sources. Do not separately list sources at the end of the writing.

If the query cannot be answered with the provided information, mention this and provide a reason why along with what it does mention.
Also cite the sources of what is actually mentioned.

Example sentences of the paragraph:
"So this is a cited sentence at the end of a paragraph[1](https://test.com). This is another sentence."
"Stephen curry is an american basketball player that plays for the warriors[1](https://www.britannica.com/biography/Stephen-Curry)."
"The economic growth forecast for the region has been adjusted from 2.5% to 3.1% due to improved trade relations[1](https://economictimes.com), while inflation rates are expected to remain steady at around 1.7% according to financial analysts[2](https://financeworld.com)."
"""
