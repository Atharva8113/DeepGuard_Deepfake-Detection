document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const modelSelect = document.getElementById("modelSelect");
    const detectBtn = document.getElementById("detectBtn");
    const resultText = document.getElementById("resultText");
    const probDetails = document.getElementById("probDetails");
    const imagePreview = document.getElementById("imagePreview");
    const detectionResults = document.getElementById("detectionResults");
    const confidenceBar = document.getElementById("confidenceBar");

    // Show preview when file selected
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        if (file) {
            imagePreview.src = URL.createObjectURL(file);
            imagePreview.style.display = "block";
        }
    });

    detectBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert("Please upload an image file.");
            return;
        }

        const model = modelSelect.value;
        const formData = new FormData();
        formData.append("image", file);
        formData.append("model", model);

        // Reset UI
        resultText.textContent = "🔍 Processing...";
        resultText.className = "processing-text"; // Changed this line to add a specific class
        probDetails.textContent = "";
        confidenceBar.style.width = "0%";
        confidenceBar.textContent = "0%";
        confidenceBar.className = "progress-bar";
        detectionResults.classList.remove("hidden", "fade-in");

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });

            const rawText = await response.text();
            console.log("📨 Raw server response:", rawText);

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            const data = JSON.parse(rawText);

            if (!data.prediction || typeof data.confidence !== "number") {
                throw new Error("Incomplete prediction data.");
            }

            // Set prediction text with glow
            if (data.prediction.toLowerCase() === "real") {
                resultText.className = "result-real glow-real";
            } else {
                resultText.className = "result-fake glow-fake";
            }
            resultText.textContent = `✅ Prediction: ${data.prediction.toUpperCase()} (Confidence: ${data.confidence.toFixed(2)}%)`;

            // Smooth animated progress bar
            const confPercent = Math.min(100, Math.max(0, data.confidence));
            let currentWidth = 0;
            const barClass = data.prediction.toLowerCase() === "real" ? "bar-real" : "bar-fake";
            confidenceBar.classList.add(barClass);

            const step = Math.max(0.5, confPercent / 50);
            const animateBar = setInterval(() => {
                if (currentWidth >= confPercent) {
                    clearInterval(animateBar);
                    confidenceBar.textContent = `${confPercent.toFixed(1)}%`;
                } else {
                    currentWidth += step;
                    confidenceBar.style.width = `${Math.min(currentWidth, confPercent)}%`;
                    confidenceBar.textContent = `${Math.min(currentWidth, confPercent).toFixed(0)}%`;
                }
            }, 15);

            // Softmax details
            if (data.probs && data.probs.length === 2) {
                probDetails.textContent =
                    `🔢 Softmax Scores → Fake: ${(data.probs[0] * 100).toFixed(2)}%, Real: ${(data.probs[1] * 100).toFixed(2)}%`;
            }

            // Fade-in effect
            requestAnimationFrame(() => {
                detectionResults.classList.add("fade-in");
            });

        } catch (err) {
            console.error("❌ Error in detection:", err);
            resultText.textContent = "❌ An error occurred during detection.";
            resultText.className = "result-fake glow-fake";
            probDetails.textContent = "";
        }
    });
});