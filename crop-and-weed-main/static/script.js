// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const segmentButton = document.getElementById('segmentButton');
    const inputCanvas = document.getElementById('inputCanvas');
    const outputCanvas = document.getElementById('outputCanvas');
    const inputPlaceholder = document.getElementById('inputPlaceholder');
    const outputPlaceholder = document.getElementById('outputPlaceholder');
    const thresholdInput = document.getElementById('thresholdInput');
    const thresholdValue = document.getElementById('thresholdValue');
    const algorithmSelect = document.getElementById('algorithmSelect');
    const ctxInput = inputCanvas.getContext('2d');
    const ctxOutput = outputCanvas.getContext('2d');
    let uploadedFile = null;

    // --- Threshold Slider Update ---
    thresholdInput.addEventListener('input', () => {
        thresholdValue.textContent = `${thresholdInput.value}%`;
    });

    // --- Image Upload Handler ---
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            uploadedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    displayImage(img);
                    segmentButton.disabled = false;
                    outputPlaceholder.style.display = 'block';
                    outputCanvas.style.display = 'none';
                    ctxOutput.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // --- Display Image on Input Canvas ---
    function displayImage(img) {
        // Set canvas size to the image size for best quality
        inputCanvas.width = outputCanvas.width = img.width;
        inputCanvas.height = outputCanvas.height = img.height;
        
        ctxInput.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        ctxInput.drawImage(img, 0, 0);
        
        inputCanvas.style.display = 'block';
        inputPlaceholder.style.display = 'none';
    }

    // --- Segmentation API Call ---
    segmentButton.addEventListener('click', async () => {
        if (!uploadedFile) {
            alert('Please upload an image first.');
            return;
        }

        segmentButton.textContent = 'Processing...';
        segmentButton.disabled = true;
        
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('model', algorithmSelect.value);
        formData.append('sensitivity', thresholdInput.value);

        try {
            const response = await fetch('/segment', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                // Display the segmented image returned as a base64 string
                const segmentedImageBase64 = data.segmented_image;
                const img = new Image();
                img.onload = () => {
                    // Draw the segmented image onto the output canvas
                    ctxOutput.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                    ctxOutput.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
                    outputCanvas.style.display = 'block';
                    outputPlaceholder.style.display = 'none';
                };
                img.src = `data:image/png;base64,${segmentedImageBase64}`;

            } else {
                alert(`Error: ${data.error || 'Unknown server error'}`);
                outputCanvas.style.display = 'none';
                outputPlaceholder.style.display = 'block';
            }

        } catch (error) {
            console.error('API Call Failed:', error);
            alert('Network error or server failed to respond.');
        } finally {
            segmentButton.textContent = '🔬 Run Segmentation';
            segmentButton.disabled = false;
        }
    });
});
