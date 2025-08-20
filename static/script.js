document.getElementById('summarize-btn').addEventListener('click', async () => {
    const text = document.getElementById('text-input').value.trim();
    const summaryDiv = document.getElementById('summary');

    if (!text) {
        summaryDiv.innerText = "⚠️ Please enter some text!";
        return;
    }

    summaryDiv.innerText = "⏳ Summarizing... Please wait.";

    try {
        const response = await fetch('/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        summaryDiv.innerText = data.summary || "⚠️ No summary generated.";
    } catch (error) {
        console.error('Error:', error);
        summaryDiv.innerText = "❌ Error calling API!";
    }
});
