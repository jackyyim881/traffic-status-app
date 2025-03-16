async function sendQuery() {
  const query = document.getElementById("queryInput").value;
  const responseElement = document.getElementById("response");

  if (!query.trim()) {
    responseElement.textContent = "Query cannot be empty!";
    return;
  }

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: query }),
    });

    const data = await response.json();

    if (data.status === "success") {
      responseElement.textContent = JSON.stringify(data, null, 2);
    } else {
      responseElement.textContent = "Error: " + data.message;
    }
  } catch (error) {
    responseElement.textContent = "Request failed: " + error.message;
  }
}
