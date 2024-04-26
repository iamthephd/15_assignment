let model;
const webcamElement = document.getElementById('webcam');
const fpsDisplay = document.getElementById('fpsValue');
const predictionsDisplay = document.getElementById('predictionsValue');


async function setupWebcam() {
    return new Promise((resolve, reject) => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                webcamElement.srcObject = stream;
                webcamElement.addEventListener('loadeddata',  () => resolve(), false);
            })
            .catch(reject);
    });
}

async function loadModel() {
    model = await tf.loadLayersModel('/saved_model/tfjs_model/model.json');
}

async function predict() {
    const startTime = performance.now();
    try {
        const img = tf.browser.fromPixels(webcamElement)
            .resizeNearestNeighbor([48, 48])
            .toFloat();
        
        // // Convert the image to grayscale by averaging the color channels
        // const grayscaleImg = img.mean(2, true);

        // // Normalize the image by dividing by 255
        const normalizedImg = img.div(255.0);

        // Add the batch dimension
        const batchedImg = normalizedImg.expandDims(0);

        const predictions = await model.predict(batchedImg);
        const predictionArray = await predictions.data();

        updatePredictions(predictionArray); // Update the prediction list display
        // Dispose the tensors to free up GPU memory
        img.dispose();
        // grayscaleImg.dispose();
        normalizedImg.dispose();
        batchedImg.dispose();

    } catch (error) {
        console.error("Error during prediction:", error);
        predictionsDisplay.textContent = "Error in predictions";
    }
    const endTime = performance.now();
    const duration = endTime - startTime;
    const fps = (1000 / duration).toFixed(2);
    fpsDisplay.textContent = fps;

    requestAnimationFrame(predict);
}

function updatePredictions(predictionArray) {
    const classDict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'};
    const predictionList = document.getElementById('predictionList');
    predictionList.innerHTML = '';

    let maxIndex = predictionArray.indexOf(Math.max(...predictionArray));

    Object.keys(classDict).forEach((key) => {
        const li = document.createElement('li');
        li.textContent = classDict[key];
        if (parseInt(key) === maxIndex) {
            li.classList.add('highlight'); // Highlight the predicted class
        }
        predictionList.appendChild(li);
    });
}

async function main() {
    model = await tf.loadLayersModel('/saved_model/tfjs_model/model.json');
    await setupWebcam();
    // await loadModel();
    predict();
}

main();
