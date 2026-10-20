def calculate_score(
    funding_amount,
    revenue,
    valuation,
    employees,
    funding_round
):

    score = 0

    # Funding Amount
    if funding_amount >= 10000000:
        score += 25
    elif funding_amount >= 5000000:
        score += 20
    elif funding_amount >= 1000000:
        score += 15
    else:
        score += 8

    # Revenue
    if revenue >= 10000000:
        score += 25
    elif revenue >= 5000000:
        score += 20
    elif revenue >= 1000000:
        score += 15
    else:
        score += 8

    # Valuation
    if valuation >= 100000000:
        score += 20
    elif valuation >= 50000000:
        score += 15
    else:
        score += 10

    # Employees
    if employees >= 500:
        score += 15
    elif employees >= 100:
        score += 10
    else:
        score += 5

    # Funding Round
    round_score = {
        "Seed": 5,
        "Series A": 10,
        "Series B": 15,
        "Series C": 20,
        "Series D": 25
    }

    score += round_score.get(funding_round, 5)

    return score


def risk_level(score):

    if score >= 80:
        return "Low"

    elif score >= 60:
        return "Medium"

    else:
        return "High"


def recommendation(score):

    if score >= 80:
        return "Invest"

    elif score >= 60:
        return "Invest with Caution"

    else:
        return "Do Not Invest"