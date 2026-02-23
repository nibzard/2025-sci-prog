/**
 * Energy Consumption Predictor
 * JavaScript Frontend Logic
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultsCard = document.getElementById('results-card');

    // Sliders and their display elements
    const tempSlider = document.getElementById('temperature');
    const windSlider = document.getElementById('wind_speed');
    const humiditySlider = document.getElementById('humidity');

    const tempValue = document.getElementById('temp-value');
    const windValue = document.getElementById('wind-value');
    const humidityValue = document.getElementById('humidity-value');

    // Set default date to today
    const dateInput = document.getElementById('date');
    const today = new Date().toISOString().split('T')[0];
    dateInput.value = today;

    // Auto-detect season based on date
    const seasonSelect = document.getElementById('season');
    updateSeasonFromDate(today);

    // Update date listener
    dateInput.addEventListener('change', function() {
        updateSeasonFromDate(this.value);
    });

    /**
     * Update season dropdown based on date
     */
    function updateSeasonFromDate(dateStr) {
        if (!dateStr) return;
        const date = new Date(dateStr);
        const month = date.getMonth(); // 0-11

        let season = 'spring';
        if (month >= 2 && month <= 4) season = 'spring';
        else if (month >= 5 && month <= 7) season = 'summer';
        else if (month >= 8 && month <= 10) season = 'autumn';
        else season = 'winter';

        seasonSelect.value = season;
    }

    /**
     * Update slider value displays
     */
    function updateSliderDisplays() {
        tempValue.textContent = tempSlider.value + '°C';
        windValue.textContent = windSlider.value + ' km/h';
        humidityValue.textContent = humiditySlider.value + '%';
    }

    // Slider event listeners
    tempSlider.addEventListener('input', updateSliderDisplays);
    windSlider.addEventListener('input', updateSliderDisplays);
    humiditySlider.addEventListener('input', updateSliderDisplays);

    // Initialize displays
    updateSliderDisplays();

    /**
     * Handle form submission
     */
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Collect form data
        const formData = {
            temperature: parseFloat(tempSlider.value),
            wind_speed: parseFloat(windSlider.value),
            humidity: parseFloat(humiditySlider.value),
            date: dateInput.value,
            season: seasonSelect.value,
            is_weekend: document.getElementById('is_weekend').checked
        };

        // Show loading state
        setLoadingState(true);

        try {
            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.status === 'success') {
                displayResults(data);
            } else {
                showError(data.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Connection error. Is the server running?');
            console.error('Error:', error);
        } finally {
            setLoadingState(false);
        }
    });

    /**
     * Set button loading state
     */
    function setLoadingState(loading) {
        if (loading) {
            predictBtn.classList.add('loading');
        } else {
            predictBtn.classList.remove('loading');
        }
    }

    /**
     * Display prediction results with animations
     */
    function displayResults(data) {
        // Activate results card
        resultsCard.classList.add('active');

        // Animate prediction number
        const predictionEl = document.getElementById('prediction-number');
        animateNumber(predictionEl, 0, data.prediction, 1000);

        // Update difference display
        const avgDiffEl = document.getElementById('avg-diff');
        const diffSign = data.avg_diff >= 0 ? '+' : '';
        avgDiffEl.textContent = diffSign + formatNumber(data.avg_diff);
        avgDiffEl.className = 'metric-value ' + (data.avg_diff >= 0 ? 'positive' : 'negative');

        // Update percentage difference
        const avgDiffPctEl = document.getElementById('avg-diff-pct');
        avgDiffPctEl.textContent = diffSign + data.avg_diff_pct + '%';

        // Update average consumption
        document.getElementById('avg-consumption').textContent = formatNumber(data.avg_consumption);

        // Update confidence
        document.getElementById('confidence').textContent = data.confidence + '%';

        // Update model R²
        document.getElementById('model-r2').textContent = data.model_r2.toFixed(2);

        // Update comparison bar
        updateComparisonBar(data.prediction, data.avg_consumption);

        // Scroll to results on mobile
        if (window.innerWidth <= 900) {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    /**
     * Animate number counting up
     */
    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        const range = end - start;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function (ease-out)
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (range * easeOut);

            element.textContent = formatNumber(Math.round(current));

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    /**
     * Format number with commas
     */
    function formatNumber(num) {
        return Math.round(num).toLocaleString('en-US');
    }

    /**
     * Update the comparison bar visualization
     */
    function updateComparisonBar(prediction, average) {
        const maxValue = Math.max(prediction, average) * 1.1;

        const avgBar = document.getElementById('bar-avg');
        const predBar = document.getElementById('bar-prediction');

        // Reset bars
        avgBar.style.width = '0%';
        predBar.style.width = '0%';

        // Animate after small delay
        setTimeout(() => {
            avgBar.style.width = (average / maxValue * 100) + '%';
            predBar.style.width = (prediction / maxValue * 100) + '%';
        }, 100);
    }

    /**
     * Show error toast notification
     */
    function showError(message) {
        const toast = document.getElementById('error-toast');
        const toastMessage = document.getElementById('toast-message');

        toastMessage.textContent = message;
        toast.classList.add('show', 'error');

        setTimeout(() => {
            toast.classList.remove('show', 'error');
        }, 4000);
    }

    /**
     * Add keyboard shortcut (Enter to predict)
     */
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.target.tagName !== 'BUTTON') {
            if (!predictBtn.classList.contains('loading')) {
                form.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Make initial prediction on page load with default values
    setTimeout(() => {
        form.dispatchEvent(new Event('submit'));
    }, 500);
});
