const dashboard = {
    overlay: null,

    init: function() {
        this.overlay = document.getElementById('dashboardOverlay');
        
        // 1. Проверка существования элемента (чтобы скрипт не падал)
        if (!this.overlay) {
            console.error("CRITICAL: Элемент #dashboardOverlay не найден! Убедитесь, что HTML модалки присутствует на странице.");
            return;
        }

        console.log("Dashboard initialized successfully");

        // 2. Обработчики событий
        // Закрытие по клику на затемненный фон
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.close();
        });

        // Закрытие по клавише Escape
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
        
        // Если данных нет (например, страница только открылась), ставим заглушки или выходим
        if (!data || !data.charts) {
            console.log("Нет реальных данных, используем заглушки");
            // Тут можно вызвать drawStubChart...
            return;
        }

        // Обновляем KPI
        if(document.getElementById('kpi-rows')) document.getElementById('kpi-rows').textContent = data.total_rows;
        if(document.getElementById('kpi-churn')) document.getElementById('kpi-churn').textContent = data.churn_rate + '%';

        setTimeout(() => {
            
            // 1. Churn Ratio
            this.createChart('chart1_Ratio', 'bar', {
                labels: data.charts.ratio.labels,
                datasets: [{
                    label: 'Клиенты',
                    data: data.charts.ratio.data,
                    backgroundColor: ['#4CAF50', '#AB00EA'],
                    borderWidth: 0
                }]
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
                        backgroundColor: '#AB00EA'
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
                        backgroundColor: '#AB00EA'
                    }
                ]
            }, { scales: { x: { stacked: true }, y: { stacked: true } } });

            // 4. Calls (Кол-во ушедших)
            this.createChart('chart4_Calls', 'bar', { // Или 'line'
                labels: data.charts.calls.labels,
                datasets: [{
                    label: 'Кол-во ушедших клиентов',
                    data: data.charts.calls.data,
                    backgroundColor: '#AB00EA',
                    borderColor: '#AB00EA',
                    borderWidth: 1
                }]
            }, {
                scales: {
                    x: { title: { display: true, text: 'Кол-во звонков' } },
                    y: { title: { display: true, text: 'Клиентов в оттоке' } }
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
