const socket = io();
 
const statusText = document.getElementById("status-text");
 
const toggleBtn = document.getElementById("toggle-btn");
const cameraBtn = document.getElementById("camera-btn");
const themeBtn = document.querySelector(".theme-btn");
 
const webcam = document.getElementById("webcam");
const carViewer = document.getElementById("car-viewer");
 
let mode = "LIVE";
let seriousTheme = false;
 
// socket connection
socket.on("connect", () => {
    console.log("Connected to GUI backend");
});
 
//status updatess
socket.on("status_update", (data) => {
    if (statusText && data.status) {
        statusText.innerText = data.status;
    }
});
 
// gsture image stream
socket.on("image_update", (data) => {
    if (!webcam) {
        console.log("webcam element not found");
        return;
    }
 
    if (!data.image) {
        console.log("No gesture image data received");
        return;
    }
 
    webcam.src = "data:image/jpeg;base64," + data.image;
});
 
//car / environment image stream
socket.on("car_image_update", (data) => {
    if (!carViewer) {
        console.log("car-viewer element not found");
        return;
    }
 
    if (!data.image) {
        console.log("No car image data received");
        return;
    }
 
    carViewer.src = "data:image/jpeg;base64," + data.image;
});
 
//gesture and command updates
socket.on("gesture_update", (data) => {
    if (!statusText) return;
 
    statusText.innerText =
        "Gesture: " + data.gesture + " | Command: " + data.command;
});
 
socket.on("confidence_update", (data) => {
    console.log("Confidence:", data.confidence);
});
 
socket.on("driving_status_update", (data) => {
    console.log("Driving status:", data.status);
});
 
socket.on("reward_update", (data) => {
    console.log("Reward:", data.reward);
});
 
//theme toggle
if (themeBtn) {
    themeBtn.addEventListener("click", () => {
        seriousTheme = !seriousTheme;
 
        if (seriousTheme) {
            document.body.classList.add("serious-theme");
            themeBtn.innerText = "THEME";
        } else {
            document.body.classList.remove("serious-theme");
            themeBtn.innerText = "THEME";
        }
 
        console.log("Theme toggled. Serious mode:", seriousTheme);
    });
} else {
    console.log("Theme button not found");
}
 
//ui updates
function updateUI() {
    if (!toggleBtn || !cameraBtn) return;
 
    toggleBtn.innerText = mode;
 
    if (mode === "LIVE") {
        cameraBtn.classList.add("disabled");
    } else {
        cameraBtn.classList.remove("disabled");
    }
}
 
//mode toggle
if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
        mode = (mode === "LIVE") ? "UPLOAD" : "LIVE";
        updateUI();
    });
}
 
//upload input setup
const uploadInput = document.createElement("input");
uploadInput.type = "file";
uploadInput.accept = "image/*";
uploadInput.style.display = "none";
document.body.appendChild(uploadInput);
 
if (cameraBtn) {
    cameraBtn.addEventListener("click", () => {
        if (mode !== "UPLOAD") return;
        uploadInput.click();
    });
}
 
//handle file upload
uploadInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;
 
    const reader = new FileReader();
 
    reader.onload = (e) => {
        const base64Image = e.target.result;
 
        socket.emit("upload_image", {
            image: base64Image
        });
    };
 
    reader.readAsDataURL(file);
});
 
updateUI();