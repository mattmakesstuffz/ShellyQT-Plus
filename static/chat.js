const form = document.querySelector("#chat-form");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");

function addMessage(text, who) {
    const bubble = document.createElement("div");
    bubble.className = `message ${who}`;
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message || send.disabled) return;

    addMessage(message, "user");
    input.value = "";
    send.disabled = true;

    const typing = addMessage("Shelly is thinking...", "shelly typing");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        typing.textContent = data.reply || "Shelly returned an empty reply.";
        typing.classList.remove("typing");
    } catch (error) {
        typing.textContent = "Couldn't connect to Shelly. Check the terminal.";
        typing.classList.remove("typing");
    } finally {
        send.disabled = false;
        input.focus();
    }
});

input.focus();
