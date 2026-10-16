
document.addEventListener("DOMContentLoaded", function () {
    const data = window.FRAUD_CHART_DATA;

    if (!data) {
        console.error("Chart data missing. Check app.py and index.html.");
        return;
    }

    // FraudGuard AI theme
    Chart.defaults.color = "#b9bac2";
    Chart.defaults.font.family = "'Segoe UI', Arial, sans-serif";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 18;

    const gridColor = "rgba(255,255,255,0.07)";
    const red = "#ed2338";
    const grey = "#92949e";

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom"
            },
            tooltip: {
                backgroundColor: "#202126",
                titleColor: "#ffffff",
                bodyColor: "#eeeeef",
                borderColor: "#414149",
                borderWidth: 1,
                padding: 12
            }
        }
    };

    // 1. Transaction Class Distribution — Doughnut chart
    const classCanvas = document.getElementById("classChart");

    if (classCanvas && data.class_chart) {
        new Chart(classCanvas, {
            type: "doughnut",
            data: {
                labels: data.class_chart.labels,
                datasets: [{
                    data: data.class_chart.values,
                    backgroundColor: [red, grey],
                    borderColor: "#151619",
                    borderWidth: 5,
                    hoverOffset: 8
                }]
            },
            options: {
                ...commonOptions,
                cutout: "70%",
                plugins: {
                    ...commonOptions.plugins,
                    legend: {
                        position: "bottom"
                    }
                }
            }
        });
    }

    // 2. Transaction Amount Distribution — Fraud vs legitimate
    const amountCanvas = document.getElementById("amountChart");

    if (amountCanvas && data.amount_chart) {
        new Chart(amountCanvas, {
            type: "bar",
            data: {
                labels: data.amount_chart.labels,
                datasets: [
                    {
                        label: "Legitimate",
                        data: data.amount_chart.legitimate,
                        backgroundColor: grey,
                        borderRadius: 3
                    },
                    {
                        label: "Fraud",
                        data: data.amount_chart.fraud,
                        backgroundColor: red,
                        borderRadius: 3
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        stacked: false,
                        grid: { display: false },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 0,
                            autoSkip: true
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    }

    // 3. Transactions by elapsed hour
    const timeCanvas = document.getElementById("timeChart");

    if (timeCanvas && data.time_chart) {
        new Chart(timeCanvas, {
            type: "line",
            data: {
                labels: data.time_chart.labels,
                datasets: [
                    {
                        label: "Legitimate",
                        data: data.time_chart.legitimate,
                        borderColor: grey,
                        backgroundColor: "rgba(146,148,158,0.10)",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2
                    },
                    {
                        label: "Fraud",
                        data: data.time_chart.fraud,
                        borderColor: red,
                        backgroundColor: "rgba(237,35,56,0.10)",
                        fill: false,
                        tension: 0.3,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Elapsed hour"
                        },
                        grid: { display: false },
                        ticks: {
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: "Transaction count"
                        },
                        grid: { color: gridColor },
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });
    }

    console.log("FraudGuard AI charts loaded.");
});
