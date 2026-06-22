import pandas as pd

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from retriever import retrieve_context

from KPIs.KPI_tools import (
    prepare_spend_df,
    prepare_budget_df,
    get_metric
)


# =========================================================
# LOAD ALL FILES
# =========================================================

spend_df = prepare_spend_df(
    pd.read_excel(
        "data/spend report for copilot.xlsx",
        header=2
    )
)

budget_df = prepare_budget_df(
    pd.read_excel(
        "data/Manpower&Electricity.xlsb",
        sheet_name="Budget",
        header=0,
        engine="pyxlsb"
    )
)

# lov_df = pd.read_excel(
#     "data/LOV-Categories.xlsx"
# )
# lov_df.columns = lov_df.columns.astype(str).str.strip()
# lov_df = lov_df.fillna("")

DATA_SOURCES = {
    "spend": spend_df,
    "budget": budget_df,
    # "lov": lov_df
}

print("\n========== DATA LOADED ==========")
print("Spend columns:", list(spend_df.columns))
print("Budget columns:", list(budget_df.columns))
# print("LOV columns:", list(lov_df.columns))
print("=================================\n")


# =========================================================
# TOOL
# =========================================================

VALID_METRICS = [
    "cost_per_sq_ft",
    "avg_cost_per_sq_ft",
    "budget_per_sq_feet",
    "over_spend",
    "sbua",
    "sum_bpcl_cost",
    "total_budget"
]

@tool
def ask_kpi(
    metric: str,
    centre_code: str = None,
    year: int = None,
    month: str = None,
    category_level1: str = None,
    category_level2: str = None
):
    """
    Use this tool to fetch a single KPI metric value from the data.
    Call this tool MULTIPLE TIMES (once per metric) if multiple metrics are needed.
    Never pass multiple metrics as a comma-separated string.

    Available metrics (use exactly as written):
    - cost_per_sq_ft
    - avg_cost_per_sq_ft
    - budget_per_sq_feet
    - over_spend
    - sbua
    - sum_bpcl_cost
    - total_budget

    Optional filters:
    - centre_code: e.g. "AMD01"
    - year: integer e.g. 2025
    - month: e.g. "January" — omit if all months needed
    - category_level1: e.g. "Hard Services"
    - category_level2: e.g. "Cleaning"

    For spend metrics (all except total_budget):
    - centre_code maps to PO Summary[Centre Code]
    - year maps to [SumYear]
    - month maps to PO Summary[Month]
    - category_level1 maps to PO Summary[CategoryLevel1]
    - category_level2 maps to PO Summary[CategoryLevel2]

    For total_budget:
    - centre_code maps to Centre Code
    - year maps to Date year
    """
    metric = metric.strip().lower()

    # Guard: reject comma-separated metrics
    if "," in metric:
        return {
            "error": (
                f"Invalid metric '{metric}'. "
                "Call this tool once per metric. "
                f"Valid metrics: {VALID_METRICS}"
            )
        }

    if metric not in VALID_METRICS:
        return {
            "error": (
                f"Unknown metric: '{metric}'. "
                f"Valid metrics are: {VALID_METRICS}"
            )
        }

    filters = {}

    # Sanitize month — reject "all", "all months", etc.
    if month and month.strip().lower() in ["all", "all months", "every month", "each month"]:
        month = None

    if metric in [
        "cost_per_sq_ft",
        "avg_cost_per_sq_ft",
        "budget_per_sq_feet",
        "over_spend",
        "sbua",
        "sum_bpcl_cost"
    ]:
        if centre_code:
            filters["PO Summary[Centre Code]"] = centre_code.strip().upper()
        if year is not None:
            filters["[SumYear]"] = year
        if month:
            filters["PO Summary[Month]"] = month
        if category_level1:
            filters["PO Summary[CategoryLevel1]"] = category_level1
        if category_level2:
            filters["PO Summary[CategoryLevel2]"] = category_level2

    elif metric == "total_budget":
        if centre_code:
            filters["Centre Code"] = centre_code.strip().upper()
        if year is not None:
            filters["Date"] = year

    return get_metric(
        data_sources=DATA_SOURCES,
        metric=metric,
        filters=filters
    )


# =========================================================
# MODEL
# =========================================================

llm = ChatOllama(
    model="qwen3:14b",
    temperature=0
)

llm_with_tools = llm.bind_tools([ask_kpi])

SYSTEM_PROMPT = SystemMessage(content="""
Before selecting a KPI, use the retrieved context.

The retrieved context contains KPI names,
business terms,
documentation,
formula explanations,
and aliases.

Use the retrieved context to identify the correct metric.
You are a KPI analyst assistant for real estate / facility management data.

You have access to a tool called `ask_kpi` which fetches ONE metric at a time from the data.

=== WHEN TO USE THE TOOL ===
Use the tool whenever the user asks for actual data values, analysis, or comparisons involving:
- total_budget
- cost_per_sq_ft
- avg_cost_per_sq_ft
- budget_per_sq_feet
- over_spend
- sbua
- sum_bpcl_cost

=== IMPORTANT RULES ===
1. NEVER pass multiple metrics as a single comma-separated string like "total_budget,cost_per_sq_ft".
   Instead, call the tool MULTIPLE TIMES — once per metric.

2. If the user asks to "analyse" a centre or says "tell me everything" or says "idk / you decide",
   fetch ALL relevant metrics automatically without asking:
   cost_per_sq_ft, avg_cost_per_sq_ft, budget_per_sq_feet, over_spend, sbua, sum_bpcl_cost, total_budget.

3. NEVER pass month as "all" or "all months". If the user wants all months, simply omit the month filter entirely.

4. Always extract filters from the conversation context:
   - centre_code: e.g. "AMD01"
   - year: e.g. 2025
   - month: a specific month name — omit if all months
   - category_level1 / category_level2: if mentioned

=== AFTER FETCHING DATA ===
Once you have all the tool results, ALWAYS produce a full analysis summary:
- Show each metric and its value clearly
- Calculate and show totals or sums where applicable (e.g. total spend = sum of all cost metrics)
- Highlight any notable observations such as over-spend, high cost per sq ft, or budget variance
- Do NOT just list raw numbers — interpret them for the user

=== WHEN NOT TO USE THE TOOL ===
- Greetings or unrelated questions
- Definitions or explanations of what a metric means
""")


# =========================================================
# CHAT LOOP
# =========================================================
chat_history = [SYSTEM_PROMPT]

while True:

    question = input("\nAsk: ")

    if question.lower() in ["exit", "quit"]:
        break

    try:
        context, metrics = retrieve_context(question)
    except Exception as e:
        print(f"\n[Retriever Error] {e}")
        context = ""
        metrics = []

    print("\n========== RETRIEVER ==========")
    print("Metrics:", metrics)
    print("================================\n")

    user_prompt = f"""
Retrieved KPI Context:

{context}

Possible KPI Matches:

{metrics}

User Question:

{question}
"""

    chat_history.append(
        HumanMessage(content=user_prompt)
    )

    # Work on a copy so we can freely append tool messages
    # without touching chat_history until we know what to save.
    messages = chat_history.copy()

    response = llm_with_tools.invoke(messages)

    # FIX: `response.tool_calls` can be None on some ChatOllama builds.
    # Guard with `or []` to avoid a TypeError on iteration.
    if response.tool_calls or []:

        # FIX: append the AIMessage that contains the tool_call requests
        # to both `messages` (for this turn's context) AND `chat_history`
        # (so the next turn remembers the model decided to call tools).
        messages.append(response)
        chat_history.append(response)

        for tool_call in (response.tool_calls or []):

            print(
                f"\n[Tool Call] "
                f"metric={tool_call['args'].get('metric')} "
                f"| filters={tool_call['args']}"
            )

            result = ask_kpi.invoke(tool_call["args"])

            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )

            # FIX: append ToolMessages to both lists so the next turn
            # has the full tool result history available.
            messages.append(tool_msg)
            chat_history.append(tool_msg)

        final_response = llm.invoke(messages)

        print("\nAnswer:")
        print(final_response.content)

        # Save the final answer to chat_history for future turns.
        chat_history.append(final_response)

    else:

        print("\nAnswer:")
        print(response.content)

        chat_history.append(response)
