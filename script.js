const URL = "./"; // model.json, metadata.json, weights.bin must be in same folder
let model, webcam, maxPredictions;
let stopped = false;

// Counter for consecutive wafer detections
let waferFrameCount = 0;
const REQUIRED_FRAMES = 6; // how many consecutive frames needed for PASS

async function init() {
  // Load model + metadata
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";
  model = await tmImage.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Setup webcam
  const flip = true;
  webcam = new tmImage.Webcam(400, 300, flip);
  await webcam.setup();
  await webcam.play();
  window.requestAnimationFrame(loop);

  document.getElementById("webcam-container").appendChild(webcam.canvas);
}

async function loop() {
  if (stopped) return;
  webcam.update();
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  const prediction = await model.predict(webcam.canvas);

  // Prediction[0] = Class 1 (Wafer), Prediction[1] = Class 2 (No Wafer)
  const waferProb = prediction[0].probability;
  const noWaferProb = prediction[1].probability;

  if (waferProb > noWaferProb && waferProb > 0.8) {
    waferFrameCount++;
  } else {
    waferFrameCount = 0; // reset if any frame is not wafer
  }

  if (waferFrameCount >= REQUIRED_FRAMES) {
    document.getElementById("result").innerText = "PASS ✅";
    // stopped = true;
    // webcam.stop(); // stop camera
  } else {
    document.getElementById("result").innerText = "No Wafer";
  }
}

init();
