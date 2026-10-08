document.addEventListener('DOMContentLoaded', function() {
    const employeeForm = document.getElementById('employee-form');
    const predictBtn = document.getElementById('predict-btn');
    const clearFormBtn = document.getElementById('clear-form-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const predictionResult = document.getElementById('prediction-result');
    const predictionValue = document.getElementById('prediction-value');
    const probabilitySection = document.getElementById('probability-section');
    const probabilityValue = document.getElementById('probability-value');
    const explanationText = document.getElementById('explanation-text');
    
    // Clear form
    clearFormBtn.addEventListener('click', function() {
        employeeForm.reset();
        predictionResult.style.display = 'none';
    });
    
    // Form submission
    employeeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(employeeForm);
        const employeeData = {};
        
        formData.forEach((value, key) => {
            // Convert numeric strings to numbers
            if (key === 'Age' || key === 'JobLevel' || key === 'MonthlyIncome' || 
                key === 'DailyRate' || key === 'HourlyRate' || key === 'MonthlyRate' ||
                key === 'PercentSalaryHike' || key === 'StockOptionLevel' ||
                key === 'TotalWorkingYears' || key === 'YearsAtCompany' ||
                key === 'YearsInCurrentRole' || key === 'YearsSinceLastPromotion' ||
                key === 'YearsWithCurrManager' || key === 'NumCompaniesWorked' ||
                key === 'TrainingTimesLastYear' || key === 'DistanceFromHome' ||
                key === 'Education' || key === 'JobSatisfaction' ||
                key === 'EnvironmentSatisfaction' || key === 'JobInvolvement' ||
                key === 'RelationshipSatisfaction' || key === 'WorkLifeBalance' ||
                key === 'PerformanceRating') {
                employeeData[key] = parseFloat(value);
            } else {
                employeeData[key] = value;
            }
        });
        
        showLoading();
        
        try {
            const response = await fetch('/predict/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    employee_data: employeeData
                })
            });
            
            const result = await response.json();
            
            hideLoading();
            
            if (result.success) {
                displayPredictionResult(result);
            } else {
                alert('Prediction failed: ' + result.error);
            }
        } catch (error) {
            hideLoading();
            alert('Error: ' + error.message);
        }
    });
    
    function displayPredictionResult(result) {
        // Set prediction value
        predictionValue.textContent = result.prediction;
        
        // Set color based on prediction
        if (result.prediction === 'Yes') {
            predictionValue.className = 'prediction-value prediction-yes';
        } else {
            predictionValue.className = 'prediction-value prediction-no';
        }
        
        // Display probability if available
        if (result.probability !== null && result.probability !== undefined) {
            probabilitySection.style.display = 'block';
            const probabilityPercent = (result.probability * 100).toFixed(1);
            probabilityValue.textContent = probabilityPercent + '%';
        } else {
            probabilitySection.style.display = 'none';
        }
        
        // Set explanation
        if (result.prediction === 'Yes') {
            explanationText.textContent = 'The selected machine learning model classified this employee as Attrition = Yes based on the information provided.';
            if (result.probability !== null) {
                explanationText.textContent += ' The model assigned a higher predicted probability to the Attrition = Yes class.';
            }
        } else {
            explanationText.textContent = 'The selected machine learning model classified this employee as Attrition = No based on the information provided.';
        }
        
        // Show result
        predictionResult.style.display = 'block';
        
        // Scroll to result
        predictionResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function showLoading() {
        loadingOverlay.style.display = 'flex';
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Predicting...';
    }
    
    function hideLoading() {
        loadingOverlay.style.display = 'none';
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="ph ph-magic-wand"></i> Predict Attrition';
    }
});
