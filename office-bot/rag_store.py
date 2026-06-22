#rag_store.py
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

embeddings = VoyageAIEmbeddings(
    model="voyage-3",
    voyage_api_key = "pa-mJkECKqEwxls4lRjlEOfwyxzO8mcVkDjLODyHBkamDR"
)

documents = [
    {
        "metric": "cost_per_sq_ft",
        "text": """
        KPI: Cost Per Sq Ft

        Aliases:
        CPSF
        Cost Per Square Foot
        Occupancy Cost

        Formula:
        Total Spend / SBUA

        Purpose:
        Measures operational cost efficiency.
        """
    },

    {
        "metric": "avg_cost_per_sq_ft",
        "text": """
        KPI: Average Cost Per Sq Ft

        Aliases:
        Average CPSF
        Avg Cost Per Sq Ft

        Purpose:
        Average occupancy cost across data.
        """
    },

    {
        "metric": "budget_per_sq_feet",
        "text": """
        KPI: Budget Per Sq Feet

        Aliases:
        Planned Cost Per Sq Ft
        Budget Density

        Purpose:
        Budget allocation per square foot.
        """
    },

    {
        "metric": "over_spend",
        "text": """
        KPI: Over Spend

        Aliases:
        Overspend
        Budget Variance

        Formula:
        Actual Spend - Budget

        Purpose:
        Measures budget deviation.
        """
    },

    {
        "metric": "sbua",
        "text": """
        KPI: SBUA

        Aliases:
        Super Built Up Area
        Area
        Square Feet

        Purpose:
        Total usable area.
        """
    },

    {
        "metric": "sum_bpcl_cost",
        "text": """
        KPI: BPCL Cost

        Aliases:
        BPCL Spend
        Fuel Cost

        Purpose:
        Total BPCL expenditure.
        """
    },

    {
        "metric": "total_budget",
        "text": """
        KPI: Total Budget

        Aliases:
        Planned Budget
        Allocated Budget

        Purpose:
        Total approved budget.
        """
    }
]

texts = [d["text"] for d in documents]

metadatas = [
    {"metric": d["metric"]}
    for d in documents
]

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./vector_db"
)

print("Vector DB Created Successfully")