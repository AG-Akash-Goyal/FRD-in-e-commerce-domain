document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.getElementById("login-form");
    const errorMessage = document.getElementById("error-message");

    loginForm.addEventListener("submit", function (e) {
        e.preventDefault();

        // Replace this with your actual authentication logic
        const username = "Akash05";
        const password = "2247205";

        const enteredUsername = loginForm.username.value;
        const enteredPassword = loginForm.password.value;

        if (enteredUsername === username && enteredPassword === password) {
            // Successful login, redirect to another page
            window.location.href = "index.html";
        } else {
            errorMessage.textContent = "Invalid username or password.";
        }
    });
});
