# services/ingestion/mock_data.py
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
from .base import Signal

fake = Faker()

THEMES = {
    "data_loss": {
        "review_phrases": [
            "I keep losing data when switching between projects, this is a dealbreaker",
            "Data disappears every time I switch tabs. Lost 2 hours of work",
            "Project switch causes data loss. Critical bug. Switching to competitor",
            "Awful experience — unsaved changes lost on project switch",
            "Losing data constantly when switching projects. Need this fixed urgently",
            "Every time I switch projects my work disappears. Unacceptable",
        ],
        "ticket_phrases": [
            "Data not saved when switching projects",
            "Lost all changes after project tab switch",
            "Autosave not working on project switch — urgent",
            "Data loss on project switch happening daily",
            "Critical: losing unsaved work on every project switch",
            "Data disappears when switching between project views",
        ],
        "sales_phrases": [
            "Customer threatening churn due to data loss issue — $450K ARR at risk",
            "Prospect saw data loss bug in demo — lost confidence in product",
            "Acme Corp put renewal on hold citing data loss bug — $2.1M deal",
            "Three enterprise accounts flagged data loss as critical blocker",
        ],
        "market_phrases": [
            "Competitor Notion has autosave — cited as advantage in analyst reports",
            "G2 reviews: data loss is top complaint across PM tools category",
        ],
        "weight": 0.40,
    },
    "sso_auth": {
        "review_phrases": [
            "No SSO support is blocking our enterprise rollout",
            "We need SAML SSO to comply with IT policy. Feature request.",
            "Competitor has SSO, we need it for enterprise deals",
            "SSO is table stakes for enterprise software in 2026",
            "IT department rejected our purchase because no SSO support",
        ],
        "ticket_phrases": [
            "SSO integration needed for enterprise deployment",
            "SAML/Okta integration required by security team",
            "Need SSO before we can roll out to full company",
            "IT policy requires SSO — blocking company-wide adoption",
        ],
        "sales_phrases": [
            "Deal on hold — Acme Corp IT requires SSO before sign-off — $2.1M",
            "Globex blocked by SSO requirement — $800K at risk",
            "Enterprise prospect cited missing SSO as number one blocker",
            "Initech deal stalled — security team requires SAML SSO — $450K",
            "Five enterprise deals blocked by missing SSO this quarter",
        ],
        "market_phrases": [
            "Gartner report: SSO is table stakes for enterprise SaaS in 2026",
            "Competitor Monday.com launched SSO for all paid tiers Q1 2026",
            "Industry benchmark: SSO adoption at 89% among enterprise SaaS tools",
        ],
        "weight": 0.25,
    },
    "slow_dashboard": {
        "review_phrases": [
            "Dashboard takes 8 seconds to load with large datasets",
            "Performance is terrible with more than 100 items",
            "App is slow. Needs optimisation urgently",
            "Loading times are unacceptable — 10 second waits daily",
            "Performance has degraded significantly in the last month",
        ],
        "ticket_phrases": [
            "Dashboard loading slowly — 10 second wait on main view",
            "Performance degradation noticed this week by multiple users",
            "App freezes when loading large project — 500 plus items",
            "Slow load times causing productivity loss across our team",
        ],
        "sales_phrases": [
            "Prospect complained about slow dashboard during demo — nearly lost deal",
            "Customer downgrade request citing performance issues — $300K ARR",
        ],
        "market_phrases": [
            "Industry benchmark: dashboard load time over 3 seconds causes 40 percent abandonment",
            "Competitor Linear cited fast performance as key differentiator in G2 report",
        ],
        "weight": 0.20,
    },
    "bulk_export": {
        "review_phrases": [
            "Need bulk export to CSV. Manual download is extremely painful",
            "Export feature is completely missing. Huge gap for our reporting needs",
            "No way to export all data at once — very frustrating for finance team",
            "Competitor has bulk export built in. We urgently need this feature",
            "Finance team cannot use this product without proper CSV export",
            "Bulk data export is essential for our compliance reporting workflow",
            "We waste hours every week manually copying data because no export exists",
            "Missing bulk export is the number one complaint from our finance users",
        ],
        "ticket_phrases": [
            "Bulk export functionality urgently requested by finance team",
            "Need CSV export for monthly compliance reporting",
            "Can we export all project records to Excel for audit purposes",
            "Compliance team requires full data export capability for regulatory audit",
            "Need to export all project data in bulk — manual copy not scalable",
            "Finance director requesting bulk export before contract renewal",
            "Audit requirement: need to export 12 months of data to CSV",
            "Export to Excel needed for board reporting — currently impossible",
        ],
        "sales_phrases": [
            "Finance team at Globex needs bulk export before sign-off — $800K deal",
            "Compliance requirement blocking Initech purchase — need CSV export — $450K",
            "Three enterprise prospects require bulk export for regulatory compliance",
            "Umbrella Ltd legal team requires data export capability — $1.2M ARR",
            "Missing export feature caused us to lose Hooli deal to competitor — $3.5M",
            "Sales team flagged bulk export as top enterprise blocker this quarter",
        ],
        "market_phrases": [
            "Competitor Notion has bulk export — cited as key advantage in analyst reviews",
            "G2 category report: data export is top requested feature across PM tools",
            "Gartner: compliance and audit capabilities increasingly required by enterprise buyers",
            "Linear added bulk export Q1 2026 — customers now comparing our feature gap",
        ],
        "weight": 0.15,
    },
}


def generate_reviews(n: int = 200) -> list[Signal]:
    signals = []
    for _ in range(n):
        theme_key = random.choices(
            list(THEMES.keys()),
            weights=[t["weight"] for t in THEMES.values()]
        )[0]
        theme = THEMES[theme_key]
        content = random.choice(theme["review_phrases"])
        if random.random() > 0.7:
            content += f" — {fake.sentence()}"
        signals.append(Signal(
            id=str(uuid.uuid4()),
            source_type="review",
            content=content,
            metadata={
                "platform": random.choice(["G2", "App Store", "Capterra", "Trustpilot"]),
                "rating": random.choices([1, 2, 3, 4, 5],
                                         weights=[15, 20, 15, 25, 25])[0],
                "theme": theme_key,
                "author": fake.name(),
            },
            created_at=datetime.now() - timedelta(days=random.randint(1, 90)),
        ))
    return signals


def generate_tickets(n: int = 150) -> list[Signal]:
    signals = []
    for _ in range(n):
        theme_key = random.choices(
            list(THEMES.keys()),
            weights=[t["weight"] for t in THEMES.values()]
        )[0]
        theme = THEMES[theme_key]
        content = random.choice(theme["ticket_phrases"])
        signals.append(Signal(
            id=str(uuid.uuid4()),
            source_type="ticket",
            content=content,
            metadata={
                "priority": random.choices(
                    ["urgent", "high", "normal", "low"],
                    weights=[10, 25, 50, 15]
                )[0],
                "csat": random.choices([1, 2, 3, 4, 5],
                                       weights=[20, 25, 20, 20, 15])[0],
                "theme": theme_key,
                "ticket_id": f"ZD-{random.randint(10000, 99999)}",
            },
            created_at=datetime.now() - timedelta(days=random.randint(1, 60)),
        ))
    return signals


def generate_sales_signals(n: int = 60) -> list[Signal]:
    signals = []
    for _ in range(n):
        theme_key = random.choices(
            list(THEMES.keys()),
            weights=[t["weight"] for t in THEMES.values()]
        )[0]
        theme = THEMES[theme_key]
        # Only use sales_phrases if they exist for this theme
        if "sales_phrases" not in theme:
            content = random.choice(theme["review_phrases"])
        else:
            content = random.choice(theme["sales_phrases"])
        signals.append(Signal(
            id=str(uuid.uuid4()),
            source_type="sales",
            content=content,
            metadata={
                "theme": theme_key,
                "source": random.choice(["Gong call", "CRM note", "Sales email"]),
                "arr_at_risk": random.choice(
                    [220000, 450000, 800000, 1200000, 2100000, 3500000]
                ),
            },
            created_at=datetime.now() - timedelta(days=random.randint(1, 45)),
        ))
    return signals


def generate_market_signals(n: int = 40) -> list[Signal]:
    signals = []
    all_market = []
    for theme in THEMES.values():
        if "market_phrases" in theme:
            all_market.extend(theme["market_phrases"])
    for _ in range(n):
        signals.append(Signal(
            id=str(uuid.uuid4()),
            source_type="market",
            content=random.choice(all_market),
            metadata={
                "source": random.choice(
                    ["Gartner", "G2 Report", "Analyst brief", "Competitor changelog"]
                )
            },
            created_at=datetime.now() - timedelta(days=random.randint(1, 120)),
        ))
    return signals


def generate_internal_signals(n: int = 20) -> list[Signal]:
    internal_phrases = [
        "OKR Q2: Reduce churn to below 3.5% — data loss is top driver",
        "Board priority: Close 3 enterprise deals requiring SSO this quarter",
        "Engineering velocity: avg 3.2 weeks per medium feature last 6 sprints",
        "NPS goal: Improve from 32 to 45 by year end",
        "Customer success: Top 5 churn reasons — data loss is number one",
        "Finance team requires bulk export for quarterly board reporting",
        "Compliance audit next quarter — data export capability required",
        "Enterprise expansion plan requires SSO and bulk export capabilities",
        "Product strategy: prioritise enterprise compliance features in H2",
        "Revenue ops: bulk export needed for customer health score reporting",
    ]
    signals = []
    for i, phrase in enumerate(internal_phrases[:n]):
        signals.append(Signal(
            id=str(uuid.uuid4()),
            source_type="internal",
            content=phrase,
            metadata={"source": "Internal strategy doc"},
            created_at=datetime.now() - timedelta(days=random.randint(1, 30)),
        ))
    return signals


def load_all_demo_signals() -> list[Signal]:
    return (
        generate_reviews(200) +
        generate_tickets(150) +
        generate_sales_signals(60) +
        generate_market_signals(40) +
        generate_internal_signals(20)
    )