const URL = "./"; // folder where model.json, metadata.json, weights.bin are stored
let model, webcam, maxPredictions;
let stopped = false; // flag to check if webcam should stop

async function init() {
  // Load model and metadata
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";
  model = await tmImage.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Setup webcam
  const flip = true; // flip webcam horizontally
  webcam = new tmImage.Webcam(400, 300, flip);
  await webcam.setup();
  await webcam.play();
  window.requestAnimationFrame(loop);

  // Attach webcam canvas to page
  document.getElementById("webcam-container").appendChild(webcam.canvas);
}

async function loop() {
  if (stopped) return; // stop loop when PASS detected
  webcam.update();
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  const prediction = await model.predict(webcam.canvas);
  let waferDetected = false;

  prediction.forEach(p => {
    if (p.className.toLowerCase().includes("wafer") && p.probability > 0.8) {
      waferDetected = true;
    }
  });

  if (waferDetected) {
    document.getElementById("result").innerText = "PASS ✅";
    stopped = true;

    // Stop webcam stream
    webcam.stop();
  } else {
    document.getElementById("result").innerText = "No Wafer";
  }
}

// Start everything
init();
