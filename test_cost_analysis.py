
from src.cost_analysis import calculate_business_cost

# Made-up test data — not real model predictions
y_true = [0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0]
amounts = [50, 100, 75, 20, 40]

results = calculate_business_cost(
    y_true=y_true,
    y_pred=y_pred,
    amounts=amounts
)

print("\nReturned results:")
print(results)