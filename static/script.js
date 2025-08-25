document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const textInput = document.getElementById('text-input');
    const actionButtonsContainer = document.getElementById('action-buttons-container');
    const stopButtonContainer = document.getElementById('stop-button-container');
    const stopButton = document.getElementById('stop-button');
    const resultSection = document.getElementById('result-section');
    const spinner = document.getElementById('spinner');

    let abortController;

    // --- UI State Management ---
    const setUiLoading = (isLoading) => {
        spinner.style.display = isLoading ? 'flex' : 'none';
        actionButtonsContainer.style.display = isLoading ? 'none' : 'flex';
        stopButtonContainer.style.display = isLoading ? 'flex' : 'none';
        textInput.disabled = isLoading;
    };

    // --- Main Handler for API Calls ---
    const handleGeneration = async (task, targetParagraph = null) => {
        const text = textInput.value;
        if (!text.trim()) {
            alert("Please enter some text first.");
            return;
        }

        if (!targetParagraph) {
            setUiLoading(true);
            resultSection.innerHTML = ''; 
        } else {
            targetParagraph.textContent = ''; 
        }
        
        abortController = new AbortController();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, task: task }),
                signal: abortController.signal,
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            if (!targetParagraph) {
                resultSection.style.display = 'block';
                targetParagraph = createResultCard(task);
            }

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        // --- FIX: Change 'return' to 'break' to allow finally block to run ---
                        if (data.trim() === '[END_OF_STREAM]') {
                            break; 
                        }
                        if (data.trim() === '[ERROR]') {
                            throw new Error('Server-side error.');
                        }
                        targetParagraph.textContent += data;
                    }
                }
                // Check again for the end-of-stream signal to exit the outer while loop
                if (lines.some(line => line.includes('[END_OF_STREAM]'))) {
                    break;
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                resultSection.innerHTML = '<p>An error occurred. Please try again.</p>';
                console.error('Generation error:', error);
            }
        } finally {
            setUiLoading(false); 
        }
    };

    // --- Helper Function to Create the Result Card ---
    const createResultCard = (task) => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const p = document.createElement('p');
        p.className = 'result-card-text';
        const isInputArabic = /[\u0600-\u06FF]/.test(textInput.value);
        if (task === 'translate') {
            p.classList.toggle('arabic-text', !isInputArabic);
        } else {
            p.classList.toggle('arabic-text', isInputArabic);
        }

        const actions = document.createElement('div');
        actions.className = 'result-card-actions';

        const regenerateBtn = document.createElement('button');
        regenerateBtn.className = 'regenerate-btn';
        regenerateBtn.title = 'Generate another version';
        regenerateBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0011.664 0l3.181-3.183m-4.991-2.69v4.992h-4.992m0 0l-3.182-3.182a8.25 8.25 0 0111.664 0l3.182 3.182" /></svg>`;
        
        regenerateBtn.addEventListener('click', () => handleGeneration(task, p));
        
        actions.appendChild(regenerateBtn);
        card.appendChild(p);
        card.appendChild(actions);
        resultSection.appendChild(card);

        return p;
    };

    // --- Event Listeners ---
    actionButtonsContainer.addEventListener('click', (e) => {
        if (e.target.matches('.action-btn')) {
            handleGeneration(e.target.dataset.task);
        }
    });

    stopButton.addEventListener('click', () => {
        if (abortController) abortController.abort();
        setUiLoading(false);
    });
});
