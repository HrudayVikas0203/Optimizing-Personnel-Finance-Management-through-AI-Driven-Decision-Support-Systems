def generate_advice(income: float, savings: float, expenses_dict: dict) -> str:
    """
    Deterministic, structured financial advice.
    No GPT-2 randomness — only data-driven rules.
    """

    total_expenses = sum(expenses_dict.values())
    savings_ratio = (savings / (income + 1)) * 100 if income > 0 else 0.0
    expense_ratio = (total_expenses / (income + 1)) * 100 if income > 0 else 0.0

    advice_sections = []

    # 1. Income vs. Savings
    advice_sections.append(f"💰 You save about **{savings_ratio:.1f}%** of your income "
                           f"and spend about **{expense_ratio:.1f}%**.")

    # 2. Identify top spending categories
    top_exp = sorted(expenses_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_exp = [f"- {k}: {v}" for k, v in top_exp[:5] if v > 0]
    if top_exp:
        advice_sections.append("📊 Your top spending categories are:\n" + "\n".join(top_exp))

    # 3. Overspending alerts
    overspending = [k for k, v in expenses_dict.items()
                    if total_expenses > 0 and (v / total_expenses) > 0.15]
    if overspending:
        advice_sections.append("⚠️ Overspending detected in: " + ", ".join(overspending) +
                               ". Try reducing these by 10–20%.")

    # 4. Emergency fund goal
    target_emergency = max(3 * (total_expenses or 0), 0)
    advice_sections.append(f"🏦 Build an emergency fund of at least **{target_emergency:.0f}** "
                           f"(≈ 3 months of expenses).")

    # 5. Actionable steps
    advice_sections.append(
        "✅ Action Plan:\n"
        "- Automate transfers: move 10–20% of income into savings/investments every month.\n"
        "- Cap lifestyle/entertainment costs to ≤ 5–8% of income.\n"
        "- If you have high-interest debt, prioritize paying it off (avalanche method).\n"
        "- Start/raise contributions to SIPs, mutual funds, or retirement accounts."
    )

    # Combine into one structured advice
    return "\n\n".join(advice_sections)
