document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const vqaForm = document.getElementById('vqa-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const placeholderText = previewContainer.querySelector('.placeholder-text');
    
    const answerPlaceholder = document.getElementById('answer-placeholder');
    const answerOutput = document.getElementById('answer-output');
    const answerText = document.getElementById('answer-text');

    // Handle Drag & Drop Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('drop-zone--over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drop-zone--over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            updateImagePreview(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length) {
            updateImagePreview(fileInput.files[0]);
        }
    });

    // Update Image Preview
    function updateImagePreview(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                placeholderText.style.display = 'none';
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '';
            imagePreview.style.display = 'none';
            placeholderText.style.display = 'block';
            placeholderText.textContent = 'Invalid file format. Please upload an image.';
        }
    }

    // Submit Form to Flask Backend
    vqaForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        const question = document.getElementById('question-input').value.trim();
        
        if (!file) {
            alert('Please select or drag an image first!');
            return;
        }
        
        if (!question) {
            alert('Please type a question!');
            return;
        }

        // Set Loading State
        submitBtn.disabled = true;
        btnLoader.style.display = 'inline-block';
        submitBtn.querySelector('span').textContent = 'Analyzing...';
        
        // Show loading placeholder in the answer block
        answerOutput.style.display = 'none';
        answerPlaceholder.style.display = 'block';
        answerPlaceholder.querySelector('p').textContent = 'Computing visual embeddings and generating response...';

        const formData = new FormData();
        formData.append('image', file);
        formData.append('question', question);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.answer) {
                // Show Answer
                answerPlaceholder.style.display = 'none';
                answerOutput.style.display = 'block';
                answerText.textContent = data.answer;
            } else {
                throw new Error(data.error || 'Failed to get a response from the model.');
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            answerPlaceholder.style.display = 'block';
            answerPlaceholder.querySelector('p').innerHTML = `<span style="color:#ef4444;">Error: ${error.message}</span>`;
            answerOutput.style.display = 'none';
        } finally {
            // Reset Button State
            submitBtn.disabled = false;
            btnLoader.style.display = 'none';
            submitBtn.querySelector('span').textContent = 'Ask Model';
        }
    });
});
