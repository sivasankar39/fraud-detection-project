
/*
===============================================
 FraudGuard AI — Overview Dashboard Charts
===============================================

Charts:
1. Transaction Class Distribution
2. Transaction Amount Distribution
3. Transaction Activity Over Time

Expected data:
window.FRAUD_CHART_DATA = {
    class_chart: {
        labels: [],
        values: []
    },
    amount_chart: {
        labels: [],
        legitimate: [],
        fraud: []
    },
    time_chart: {
        labels: [],
        legitimate: [],
        fraud: []
    }
};
*/

document.addEventListener("DOMContentLoaded", function () {

    const data = window.FRAUD_CHART_DATA;

    // Only run on the Overview page.
    const classCanvas = document.getElementById("classChart");
    const amountCanvas = document.getElementById("amountChart");
    const timeCanvas = document.getElementById("timeChart");

    if (!classCanvas && !amountCanvas && !timeCanvas) {
        return;
    }

    if (typeof Chart === "undefined") {
        console.error(
            "FraudGuard AI: Chart.js is not loaded."
        );
        return;
    }

    if (!data || typeof data !== "object") {
        console.error(
            "FraudGuard AI: Overview chart data is missing."
        );
        return;
    }

    // =========================================
    // DASHBOARD COLORS
    // =========================================

    const COLORS = {
        blue: "#3296f5",
        red: "#ff3048",
        navy: "#172b4d",
        muted: "#64748b",
        grid: "#e9eef5",
        white: "#ffffff",
        blueFill: "rgba(50,150,245,0.10)",
        redFill: "rgba(255,48,72,0.10)"
    };

    Chart.defaults.color = COLORS.muted;
    Chart.defaults.font.family =
        "'Segoe UI', Arial, sans-serif";
    Chart.defaults.font.size = 12;

    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 18;

    // =========================================
    // COMMON OPTIONS
    // =========================================

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,

        plugins: {
            legend: {
                display: true,
                position: "bottom",
                labels: {
                    color: COLORS.muted,
                    usePointStyle: true,
                    pointStyle: "circle",
                    padding: 18
                }
            },

            tooltip: {
                backgroundColor: COLORS.navy,
                titleColor: COLORS.white,
                bodyColor: COLORS.white,
                padding: 12,
                cornerRadius: 9
            }
        }
    };

    // =========================================
    // HELPER FUNCTIONS
    // =========================================

    function validArray(value) {
        return Array.isArray(value) ? value : [];
    }

    function numericArray(value) {
        return validArray(value).map(item => {
            const number = Number(item);
            return Number.isFinite(number) ? number : 0;
        });
    }

    function createChart(canvas, config) {
        if (!canvas) return null;

        try {
            // Prevent duplicate chart instances.
            const existing = Chart.getChart(canvas);

            if (existing) {
                existing.destroy();
            }

            return new Chart(canvas, config);

        } catch (error) {
            console.error(
                "FraudGuard AI: Chart rendering failed:",
                canvas.id,
                error
            );

            return null;
        }
    }

    // =========================================
    // DONUT CENTER TEXT PLUGIN
    // =========================================

    const centerTextPlugin = {
        id: "fraudGuardCenterText",

        afterDraw(chart) {
            if (chart.config.type !== "doughnut") {
                return;
            }

            const meta = chart.getDatasetMeta(0);

            if (!meta || !meta.data.length) {
                return;
            }

            const arc = meta.data[0];

            const values = chart.data.datasets[0].data;

            const total = values.reduce(
                (sum, value) => sum + (Number(value) || 0),
                0
            );

            const ctx = chart.ctx;

            ctx.save();

            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            ctx.fillStyle = COLORS.navy;
            ctx.font =
                "bold 23px 'Segoe UI', Arial, sans-serif";

            ctx.fillText(
                total.toLocaleString("en-US"),
                arc.x,
                arc.y - 12
            );

            ctx.fillStyle = COLORS.muted;
            ctx.font =
                "11px 'Segoe UI', Arial, sans-serif";

            ctx.fillText(
                "Total Transactions",
                arc.x,
                arc.y + 17
            );

            ctx.restore();
        }
    };

    // =========================================
    // 1. TRANSACTION CLASS DISTRIBUTION
    // =========================================

    if (classCanvas && data.class_chart) {

        const labels = validArray(
            data.class_chart.labels
        );

        const values = numericArray(
            data.class_chart.values
        );

        if (
            labels.length > 0 &&
            labels.length === values.length
        ) {

            const colors = labels.map(label => {

                const name = String(label).toLowerCase();

                const isLegitimate =
                    /non[\s-]?fraud|legitimate|genuine/.test(name);

                const isFraud =
                    /fraud/.test(name) && !isLegitimate;

                return isFraud
                    ? COLORS.red
                    : COLORS.blue;
            });

            createChart(classCanvas, {
                type: "doughnut",

                data: {
                    labels: labels,

                    datasets: [{
                        data: values,
                        backgroundColor: colors,
                        borderColor: COLORS.white,
                        borderWidth: 2,
                        hoverOffset: 6
                    }]
                },

                plugins: [centerTextPlugin],

                options: {
                    ...commonOptions,
                    cutout: "72%"
                }
            });
        }
    }

    // =========================================
    // 2. TRANSACTION AMOUNT DISTRIBUTION
    // =========================================

    if (amountCanvas && data.amount_chart) {

        const amountData = data.amount_chart;

        const labels = validArray(
            amountData.labels
        );

        const legitimate = numericArray(
            amountData.legitimate
        );

        const fraud = numericArray(
            amountData.fraud
        );

        if (
            labels.length > 0 &&
            legitimate.length === labels.length &&
            fraud.length === labels.length
        ) {

            createChart(amountCanvas, {
                type: "bar",

                data: {
                    labels: labels,

                    datasets: [
                        {
                            label: "Legitimate",
                            data: legitimate,
                            backgroundColor: COLORS.blue,
                            borderRadius: 4,
                            maxBarThickness: 28
                        },
                        {
                            label: "Fraud",
                            data: fraud,
                            backgroundColor: COLORS.red,
                            borderRadius: 4,
                            maxBarThickness: 28
                        }
                    ]
                },

                options: {
                    ...commonOptions,

                    scales: {
                        x: {
                            grid: {
                                display: false
                            },

                            ticks: {
                                color: COLORS.muted,
                                maxRotation: 45,
                                minRotation: 35
                            }
                        },

                        y: {
                            beginAtZero: true,

                            grid: {
                                color: COLORS.grid
                            },

                            ticks: {
                                color: COLORS.muted,
                                precision: 0
                            }
                        }
                    }
                }
            });
        }
    }

    // =========================================
    // 3. TRANSACTION ACTIVITY OVER TIME
    // =========================================

    if (timeCanvas && data.time_chart) {

        const timeData = data.time_chart;

        const labels = validArray(
            timeData.labels
        );

        const legitimate = numericArray(
            timeData.legitimate
        );

        const fraud = numericArray(
            timeData.fraud
        );

        if (
            labels.length > 0 &&
            legitimate.length === labels.length &&
            fraud.length === labels.length
        ) {

            createChart(timeCanvas, {
                type: "line",

                data: {
                    labels: labels,

                    datasets: [
                        {
                            label: "Legitimate",
                            data: legitimate,
                            borderColor: COLORS.blue,
                            backgroundColor: COLORS.blueFill,
                            fill: true,
                            tension: 0.35,
                            pointRadius: 2,
                            borderWidth: 2.5
                        },
                        {
                            label: "Fraud",
                            data: fraud,
                            borderColor: COLORS.red,
                            backgroundColor: COLORS.redFill,
                            fill: false,
                            tension: 0.35,
                            pointRadius: 3,
                            borderWidth: 2
                        }
                    ]
                },

                options: {
                    ...commonOptions,

                    interaction: {
                        mode: "index",
                        intersect: false
                    },

                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: "Elapsed Hour",
                                color: COLORS.muted
                            },

                            grid: {
                                display: false
                            },

                            ticks: {
                                color: COLORS.muted,
                                maxTicksLimit: 12
                            }
                        },

                        y: {
                            beginAtZero: true,

                            title: {
                                display: true,
                                text: "Transaction Count",
                                color: COLORS.muted
                            },

                            grid: {
                                color: COLORS.grid
                            },

                            ticks: {
                                color: COLORS.muted,
                                precision: 0
                            }
                        }
                    }
                }
            });
        }
    }

    console.log(
        "FraudGuard AI: Overview charts initialized."
    );

});
