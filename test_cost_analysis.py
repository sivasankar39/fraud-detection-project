
from src.cost_analysis import calculate_business_cost
from src.cost_analysis import calculate_cost_across_thresholds
from pathlib import Path
from src.cost_analysis import calculate_cost_across_thresholds

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

# Small sample data for testing only
y_true = [0, 0, 1, 1]
y_prob = [0.1, 0.8, 0.6, 0.3]
amounts = [100, 200, 50, 90]

cost_results = calculate_cost_across_thresholds(
    y_true=y_true,
    y_prob=y_prob,
    amounts=amounts,
)

# Save sample results for testing only
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "sample_cost_analysis.csv"
cost_results.to_csv(output_file, index=False)

print(f"\nSample cost analysis saved to: {output_file}")

print(cost_results.to_string(index=False))

def test_cost_across_thresholds():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.8, 0.6, 0.3]
    amounts = [100, 200, 50, 90]

    result = calculate_cost_across_thresholds(
        y_true, y_prob, amounts
    )

    # At threshold 0.2
    row = result[result["threshold"] == 0.2].iloc[0]
    assert row["FP"] == 1
    assert row["FN"] == 0
    assert row["total_cost_USD"] == 2.0

    # At threshold 0.5
    row = result[result["threshold"] == 0.5].iloc[0]
    assert row["FP"] == 1
    assert row["FN"] == 1
    assert row["total_cost_USD"] == 92.0

    # At threshold 0.9
    row = result[result["threshold"] == 0.9].iloc[0]
    assert row["FP"] == 0
    assert row["FN"] == 2
    assert row["total_cost_USD"] == 140.0

    print("All threshold cost tests passed!")


if __name__ == "__main__":
    test_cost_across_thresholds()
