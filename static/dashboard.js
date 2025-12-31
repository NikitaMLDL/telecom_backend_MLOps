window.SERVER_DATA = {
    total_rows: 1000,
    churn_rate: 15.5,
    charts: {
        ratio: {
            labels: ['Лояльные', 'Отток'],
            data: [845, 155]
        },
        intl: {
            labels: ['No', 'Yes'],
            stayed: [700, 145],
            churned: [100, 55]
        },
        vmail: {
            labels: ['No', 'Yes'],
            stayed: [600, 245],
            churned: [120, 35]
        },
        calls: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9+'],
            stayed: [200, 350, 150, 80, 40, 15, 5, 3, 2, 0],
            churned: [10, 20, 30, 45, 35, 10, 4, 1, 0, 0]
        }
    }
};

var accent_color = "#AB00EA"
var loyal_color  = "#4CAF50"
var churn_color  = "#ef7e15"

const dashboard = {
    overlay: null,

    init: function() {
        this.overlay = document.getElementById('dashboardOverlay');

        if (!this.overlay) {
            console.error("Элемент #dashboardOverlay не найден!");
            return;
        }

        console.log("Dashboard initialized successfully");
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.close();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.overlay.style.display === 'flex') this.close();
        });
    },

    open: function() {
        if (!this.overlay) this.init();
        if (!this.overlay) return;

        this.overlay.classList.add('is-open');
        this.overlay.setAttribute('aria-hidden', 'false');
        document.body.classList.add('no-scroll');
        this.renderCharts();
    },

    close: function() {
        if (!this.overlay) return;

        this.overlay.classList.remove('is-open');
        this.overlay.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('no-scroll');
    },

    // Основная функция отрисовки всех 4-х графиков
    renderCharts: function() {
        // Берем данные, пришедшие с Python
        const data = window.SERVER_DATA;

        if (!data || !data.charts) {
            console.log("Нет реальных данных, используем заглушки");
            return;
        }

        // Обновляем KPI
        if(document.getElementById('kpi-rows')) document.getElementById('kpi-rows').textContent = data.total_rows;
        if(document.getElementById('kpi-churn')) document.getElementById('kpi-churn').textContent = data.churn_rate + '%';

        setTimeout(() => {
            // 1. Churn Ratio (с процентами в тултипе)
            this.createChart('chart1_Ratio', 'bar', {
                labels: data.charts.ratio.labels,
                datasets: [{
                    label: 'Клиенты',
                    data: data.charts.ratio.data,
                    backgroundColor: [loyal_color, accent_color],
                    borderWidth: 0
                }]
            }, {
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                let value = context.parsed.y;
                                // Считаем сумму всех значений в наборе данных
                                let total = context.dataset.data.reduce((a, b) => a + b, 0);
                                let percentage = Math.round((value / total) * 100) + '%';
                                return `${label}: ${value} (${percentage})`;
                            }
                        }
                    }
                }
            });

            // 2. Intl Plan
            this.createChart('chart2_Intl', 'bar', {
                labels: data.charts.intl.labels,
                datasets: [
                    {
                        label: 'Остались',
                        data: data.charts.intl.stayed,
                        backgroundColor: '#e0e0e0'
                    },
                    {
                        label: 'Ушли',
                        data: data.charts.intl.churned,
                        backgroundColor: accent_color
                    }
                ]
            }, { scales: { x: { stacked: true }, y: { stacked: true } } });

            // 3. VMail Plan
            this.createChart('chart3_VMail', 'bar', {
                labels: data.charts.vmail.labels,
                datasets: [
                    {
                        label: 'Остались',
                        data: data.charts.vmail.stayed,
                        backgroundColor: '#e0e0e0'
                    },
                    {
                        label: 'Ушли',
                        data: data.charts.vmail.churned,
                        backgroundColor: accent_color
                    }
                ]
            }, { scales: { x: { stacked: true }, y: { stacked: true } } });

            // 4. Calls (Calls vs Churn/Stay)
            this.createChart('chart4_Calls', 'bar', {
                labels: data.charts.calls.labels,
                datasets: [
                    {
                        label: 'Остались',
                        data: data.charts.calls.stayed,
                        backgroundColor: loyal_color // Серый для лояльных
                    },
                    {
                        label: 'Ушедшие',
                        data: data.charts.calls.churned,
                        backgroundColor: accent_color // Фиолетовый для оттока
                    }
                ]
            }, {
                scales: {
                    x: { 
                        title: { display: true, text: 'Кол-во звонков' },
                        stacked: false // false = столбцы рядом, true = друг на друге
                    },
                    y: { 
                        title: { display: true, text: 'Количество клиентов' },
                        stacked: false
                    }
                }
            });

        }, 50);
    },

    // Вспомогательная функция для создания графиков (убирает дублирование кода)
    createChart: function(canvasId, type, data, extraOptions = {}) {
        const canvas = document.getElementById(canvasId);
        
        // Если элемента нет в HTML, пропускаем (чтобы не было ошибок)
        if (!canvas) return;

        // Проверка: загружена ли библиотека Chart.js
        if (typeof Chart === 'undefined') {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f8f9fa'; 
            ctx.fillRect(0,0,canvas.width,canvas.height);
            ctx.fillStyle = '#dc3545'; 
            ctx.font = "14px sans-serif";
            ctx.fillText('Ошибка: Chart.js не загружен', 10, 30);
            return;
        }

        const ctx = canvas.getContext('2d');
        
        // Если на этом canvas уже есть график - уничтожаем его перед созданием нового
        const existingChart = Chart.getChart(ctx);
        if (existingChart) existingChart.destroy();

        // Базовые настройки, общие для всех
        const defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            interaction: {
                mode: 'index',
                intersect: false,
            }
        };

        // Объединяем базовые настройки с индивидуальными
        const options = { ...defaultOptions, ...extraOptions };

        // Создаем график
        new Chart(ctx, { type, data, options });
    }
};

// Автозапуск инициализации при полной загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    dashboard.init();
});
