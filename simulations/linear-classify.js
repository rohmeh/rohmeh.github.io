function initLinearClassificationDemo() {
    console.log("Initializing Linear Classification Demo...");
    const container = document.getElementById('simulationContainer');
    container.innerHTML = `
        <div class="linear-classification-container" style="max-width: 900px; margin: 0 auto;">
            <h1 class="main_color text-center" style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1.3rem;">
                Linear Classifier Demo
            </h1>
            <div class="description mb-3 p-2 rounded" style="background-color: #f8f9fa; border: 1px solid #e6e6e6; font-size: 0.85rem;">
                <p class="secondary_color">This demo visualizes how linear classifiers work. The training data is <span style="font-family: serif;">x<sub>i</sub></span> with labels <span style="font-family: serif;">y<sub>i</sub></span>. The class scores are computed as <span style="font-family: serif;">f(x<sub>i</sub>; W, b) = W x<sub>i</sub> + b</span>, where <span style="font-family: serif;">W</span> is the weight matrix and <span style="font-family: serif;">b</span> is the bias vector. Each data point <span style="font-family: serif;">x<sub>i</sub></span> is 2D and belongs to one of 3 classes. The weight matrix is of size [3×2] and bias vector is of size [3×1].</p>
                <p><strong>Loss Function (Default: Weston-Watkins SVM):</strong></p>
                <p style="text-align: center;">
                    <span style="font-family: serif;">
                        L = (1/N) ∑<sub>i</sub>∑<sub>j≠y<sub>i</sub></sub> max(0, f<sub>j</sub> − f<sub>y<sub>i</sub></sub> + 1) + λ∑W²
                    </span>
                </p>
                <p class="secondary_color">Where N is the number of examples, and λ is a hyperparameter that controls the strength of the L2 regularization penalty</p>
                <p class="secondary_color">Training points are colored by class (red, green, blue). The learning rate used is 0.1, and you can initialize the initial weights with desired values. Background shows classification regions. Each classifier is visualized by a line where the score equals zero. You can drag data points to observe how they affect the loss and decision boundaries in real time.</p>
                <p class="mb-0"><small><em>Inspired by the <a href="http://vision.stanford.edu/teaching/cs231n/" target="_blank" style="color: #8C1515;">CS231n: Convolutional Neural Networks for Visual Recognition</a> course at Stanford University.</em></small></p>
            </div>

            <div class="row">
                <div class="col-lg-7">
                    <div class="card p-1 mb-2">
                        <canvas id="lcCanvas" width="550" height="400" style="border:1px solid #e6e6e6; background:white; width: 100%;"></canvas>
                        <p class="text-muted small mt-1 text-center secondary_font">
                            Drag points to move them. Background shows classification regions.
                        </p>
                    </div>
                </div>
                <!-- Controls Panel -->
                <div class="col-lg-5">
                    <div class="card p-2 mb-2">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h4 class="main_color mb-0" style="font-size: 1rem;">Parameters</h4>
                            <div>
                                <button id="lcRandom" class="btn btn-sm btn-outline-secondary py-0">
                                    Randomize
                                </button>
                            </div>
                        </div>
                        <div class="matrix-controls mb-2" style="overflow-x: auto;">
                            <h5 style="font-size: 0.9rem;">Weight Matrix (W)</h5>
                            <table id="weightMatrix" class="table table-sm table-bordered mb-1" style="font-size: 0.8rem; min-width: 250px;">
                                <thead>
                                    <tr>
                                        <th style="width: 20%">Class</th>
                                        <th style="width: 26%">w<sub>0</sub></th>
                                        <th style="width: 26%">w<sub>1</sub></th>
                                        <th style="width: 26%">b</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <th>0</th>
                                        <td><input type="number" class="form-control form-control-sm" value="1.0" data-row="0" data-col="0" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="2.0" data-row="0" data-col="1" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="0.0" data-row="0" data-col="2" step="0.1"></td>
                                    </tr>
                                    <tr>
                                        <th>1</th>
                                        <td><input type="number" class="form-control form-control-sm" value="2.0" data-row="1" data-col="0" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="-4.0" data-row="1" data-col="1" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="0.5" data-row="1" data-col="2" step="0.1"></td>
                                    </tr>
                                    <tr>
                                        <th>2</th>
                                        <td><input type="number" class="form-control form-control-sm" value="3.0" data-row="2" data-col="0" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="-1.0" data-row="2" data-col="1" step="0.1"></td>
                                        <td><input type="number" class="form-control form-control-sm" value="-0.5" data-row="2" data-col="2" step="0.1"></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="form-group mb-2">
                            <label class="secondary_font mb-0"><strong>Regularization (λ):</strong></label>
                            <input type="range" id="lcRegC" min="0" max="0.5" step="0.001" value="0.1" class="form-control form-control-sm">
                            <small class="form-text text-muted" id="lcRegValue">Default: 0.1000</small>
                        </div>
                        <div class="form-group mb-1">
                            <label class="secondary_font mb-1"><strong>Loss Function:</strong></label>
                            <select id="lcForm" class="form-control form-control-sm">
                                <option value="ww">Weston-Watkins SVM</option>
                                <option value="ova">One vs All SVM</option>
                                <option value="ssvm">Structured SVM</option>
                                <option value="softmax">Softmax</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Loss Computation Table -->
            <div class="card mt-2">
                <div class="card-header py-1 d-flex justify-content-between align-items-center">
                    <h5 class="mb-0" style="font-size: 0.9rem;">Loss Computation</h5>
                    <div>
                        <button id="lcStep" class="btn btn-sm py-0" style="background-color: #8C1515; color: white;">Single Step</button>
                        <button id="lcStart" class="btn btn-sm py-0 ml-1" style="background-color: #8C1515; color: white;">Start</button>
                        <button id="lcStop" class="btn btn-sm py-0 ml-1 btn-outline-secondary">Stop</button>
                    </div>
                </div>
                <div class="card-body p-1" style="overflow-x: auto;">
                    <table class="table table-sm mb-0" style="font-size: 0.75rem;">
                        <thead>
                            <tr>
                                <th>Point</th>
                                <th>x[0]</th>
                                <th>x[1]</th>
                                <th style="background-color: #eee; border-left: 2px solid #ccc;">y</th>
                                <th style="background-color: #FFCDD2">s[0]</th>
                                <th style="background-color: #C8E6C9">s[1]</th>
                                <th style="background-color: #BBDEFB">s[2]</th>
                                <th style="border-left: 2px solid #ccc;">Loss</th>
                            </tr>
                        </thead>
                        <tbody id="pointValues"></tbody>
                    </table>
                </div>
            </div>

            <!-- Loss Summary -->
            <div class="text-center mt-2 mb-4">
                <div><strong>Mean Data Loss:</strong> <span id="meanDataLoss" style="font-size: 1.1rem;">0.0000</span></div>
                <div><strong>Regularization Loss (λ=<span id="lambdaValue">0.1000</span>):</strong> <span id="regLoss" style="font-size: 1.1rem;">0.0000</span></div>
                <div><strong>Total Loss:</strong> <span id="totalLoss" style="font-size: 1.2rem; color: #d9534f;">0.0000</span></div>
            </div>
        </div>
    `;

    // Constants and initial setup
    const WIDTH = 550, HEIGHT = 400, SCALE = 80, NUM_POINTS = 9;
    const COLORS = ["#FF5252", "#4CAF50", "#2196F3"];
    const LIGHT_COLORS = ["#FFCDD2", "#C8E6C9", "#BBDEFB"];
    const INITIAL_DATA = [
        [0.5, 0.4], [0.8, 0.3], [0.3, 0.8],
        [-0.4, 0.3], [-0.3, 0.7], [-0.7, 0.2],
        [0.7, -0.4], [0.75, -2.25], [-0.4, -0.5]
    ];
    const INITIAL_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const state = {
        data: JSON.parse(JSON.stringify(INITIAL_DATA)),
        labels: [...INITIAL_LABELS],
        W: [
            [1.0, 2.0, 0.0],
            [2.0, -4.0, 0.5],
            [3.0, -1.0, -0.5]
        ],
        regC: 0.1,
        lossType: "ww",
        isTraining: false,
        trainingInterval: null,
        draggingIdx: -1,
        gradW: [[0,0,0],[0,0,0],[0,0,0]]
    };

    const canvas = document.getElementById("lcCanvas");
    const ctx = canvas.getContext("2d");
    const pointValuesTable = document.getElementById("pointValues");

    function drawAll() {
        ctx.clearRect(0, 0, WIDTH, HEIGHT);
        drawDecisionRegions();
        drawAxes();
        drawDecisionBoundaries();
        drawDataPoints();
        updatePointDetails();
    }

    function drawDecisionRegions() {
        const density = 8;
        for (let x = 0; x <= WIDTH; x += density) {
            for (let y = 0; y <= HEIGHT; y += density) {
                const dx = (x - WIDTH / 2) / SCALE;
                const dy = (y - HEIGHT / 2) / SCALE;
                const scores = [
                    state.W[0][0] * dx + state.W[0][1] * dy + state.W[0][2],
                    state.W[1][0] * dx + state.W[1][1] * dy + state.W[1][2],
                    state.W[2][0] * dx + state.W[2][1] * dy + state.W[2][2]
                ];
                const winningClass = scores.indexOf(Math.max(...scores));
                ctx.fillStyle = `${LIGHT_COLORS[winningClass]}80`;
                ctx.fillRect(x - density/2, y - density/2, density, density);
            }
        }
    }

    function drawAxes() {
        ctx.strokeStyle = "#888";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, HEIGHT / 2);
        ctx.lineTo(WIDTH, HEIGHT / 2);
        ctx.moveTo(WIDTH / 2, 0);
        ctx.lineTo(WIDTH / 2, HEIGHT);
        ctx.stroke();
    }

    function drawDecisionBoundaries() {
        for (let classIdx = 0; classIdx < 3; classIdx++) {
            ctx.strokeStyle = COLORS[classIdx];
            ctx.lineWidth = 2;
            const w0 = state.W[classIdx][0];
            const w1 = state.W[classIdx][1];
            const bias = state.W[classIdx][2];
            const x0 = -5, x1 = 5;
            const y0 = (-bias - w0 * x0) / w1;
            const y1 = (-bias - w0 * x1) / w1;
            const sx0 = x0 * SCALE + WIDTH / 2;
            const sy0 = y0 * SCALE + HEIGHT / 2;
            const sx1 = x1 * SCALE + WIDTH / 2;
            const sy1 = y1 * SCALE + HEIGHT / 2;
            ctx.beginPath();
            ctx.moveTo(sx0, sy0);
            ctx.lineTo(sx1, sy1);
            ctx.stroke();
            const midX = (sx0 + sx1) / 2;
            const midY = (sy0 + sy1) / 2;
            ctx.fillStyle = COLORS[classIdx];
            ctx.font = "bold 12px Arial";
            ctx.fillText(`Class ${classIdx}`, midX + 8, midY - 8);
        }
    }

    function drawDataPoints() {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        for (let i = 0; i < NUM_POINTS; i++) {
            const x = state.data[i][0] * SCALE + WIDTH / 2;
            const y = state.data[i][1] * SCALE + HEIGHT / 2;
            ctx.fillStyle = COLORS[state.labels[i]];
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            if (i === state.draggingIdx) {
                ctx.strokeStyle = "#000";
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y, 7, 0, Math.PI * 2);
                ctx.stroke();
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1.5;
            }
        }
    }

    function computeLossAndGradient() {
        state.gradW = [[0,0,0],[0,0,0],[0,0,0]];
        let dataLoss = 0;
        const scores = [];
        const losses = [];
        for (let i = 0; i < NUM_POINTS; i++) {
            const x0 = state.data[i][0];
            const x1 = state.data[i][1];
            const label = state.labels[i];
            const s = [
                state.W[0][0] * x0 + state.W[0][1] * x1 + state.W[0][2],
                state.W[1][0] * x0 + state.W[1][1] * x1 + state.W[1][2],
                state.W[2][0] * x0 + state.W[2][1] * x1 + state.W[2][2]
            ];
            scores.push(s);
            let loss_i = 0;
            if (state.lossType === "ww" || state.lossType === "ssvm") {
                for (let j = 0; j < 3; j++) {
                    if (j === label) continue;
                    let margin = s[j] - s[label];
                    if (state.lossType === "ww") margin += 1;
                    if (margin > 0) {
                        loss_i += margin;
                        state.gradW[j][0] += x0;
                        state.gradW[j][1] += x1;
                        state.gradW[j][2] += 1;
                        state.gradW[label][0] -= x0;
                        state.gradW[label][1] -= x1;
                        state.gradW[label][2] -= 1;
                    }
                }
            } else if (state.lossType === "ova") {
                for (let j = 0; j < 3; j++) {
                    const margin = (j === label ? 1 : -1) * s[j] + 1;
                    if (margin > 0) {
                        loss_i += margin;
                        const sign = (j === label ? 1 : -1);
                        state.gradW[j][0] += sign * x0;
                        state.gradW[j][1] += sign * x1;
                        state.gradW[j][2] += sign;
                    }
                }
            } else if (state.lossType === "softmax") {
                const maxScore = Math.max(...s);
                const expScores = s.map(score => Math.exp(score - maxScore));
                const sumExp = expScores.reduce((a, b) => a + b, 0);
                const probs = expScores.map(score => score / sumExp);
                loss_i = -Math.log(probs[label]);
                for (let j = 0; j < 3; j++) {
                    const gradient = (j === label ? probs[j] - 1 : probs[j]);
                    state.gradW[j][0] += gradient * x0;
                    state.gradW[j][1] += gradient * x1;
                    state.gradW[j][2] += gradient;
                }
            }
            losses.push(loss_i);
            dataLoss += loss_i;
        }
        dataLoss /= NUM_POINTS;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                state.gradW[i][j] /= NUM_POINTS;
            }
        }
        let regLoss = 0;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 2; j++) {
                regLoss += state.regC * state.W[i][j] ** 2;
                state.gradW[i][j] += 2 * state.regC * state.W[i][j];
            }
        }
        const totalLoss = dataLoss + regLoss;
        return { scores, losses, dataLoss, regLoss, totalLoss };
    }

    function updatePointDetails() {
        const { scores, losses, dataLoss, regLoss, totalLoss } = computeLossAndGradient();
        let pointHtml = '';
        for (let i = 0; i < NUM_POINTS; i++) {
            pointHtml += `
                <tr>
                    <td>${i}</td>
                    <td>${state.data[i][0].toFixed(2)}</td>
                    <td>${state.data[i][1].toFixed(2)}</td>
                    <td style="background-color: ${COLORS[state.labels[i]]}66; color: white; font-weight: bold;">${state.labels[i]}</td>
                    <td style="background-color: #FFCDD2">${scores[i][0].toFixed(2)}</td>
                    <td style="background-color: #C8E6C9">${scores[i][1].toFixed(2)}</td>
                    <td style="background-color: #BBDEFB">${scores[i][2].toFixed(2)}</td>
                    <td>${losses[i].toFixed(2)}</td>
                </tr>
            `;
        }
        pointValuesTable.innerHTML = pointHtml;
        document.getElementById("meanDataLoss").textContent = dataLoss.toFixed(4);
        document.getElementById("regLoss").textContent = regLoss.toFixed(4);
        document.getElementById("totalLoss").textContent = totalLoss.toFixed(4);
        document.getElementById("lambdaValue").textContent = state.regC.toFixed(4);
    }

    function singleStep() {
        const { totalLoss } = computeLossAndGradient();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                state.W[i][j] -= 0.1 * state.gradW[i][j];
            }
        }
        updateParameterInputs();
        drawAll();
    }

    function startTraining() {
        if (state.isTraining) return;
        state.isTraining = true;
        document.getElementById("lcStart").disabled = true;
        document.getElementById("lcStop").disabled = false;
        state.trainingInterval = setInterval(() => {
            singleStep();
        }, 100);
    }

    function stopTraining() {
        if (!state.isTraining) return;
        clearInterval(state.trainingInterval);
        state.isTraining = false;
        document.getElementById("lcStart").disabled = false;
        document.getElementById("lcStop").disabled = true;
    }

    function randomizeWeights() {
        stopTraining();
        state.W = [
            [(Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 2],
            [(Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 2],
            [(Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 2]
        ];
        updateParameterInputs();
        drawAll();
    }

    function updateParameterInputs() {
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const input = document.querySelector(`#weightMatrix input[data-row="${i}"][data-col="${j}"]`);
                if (input) input.value = state.W[i][j].toFixed(2);
            }
        }
    }

    function updateParametersFromUI() {
        document.querySelectorAll("#weightMatrix input").forEach(input => {
            const row = parseInt(input.dataset.row);
            const col = parseInt(input.dataset.col);
            state.W[row][col] = parseFloat(input.value);
        });
    }

    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) * (canvas.width / rect.width),
            y: (evt.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    canvas.addEventListener("mousedown", (e) => {
        const mousePos = getMousePos(canvas, e);
        let minDist = Infinity;
        for (let i = 0; i < NUM_POINTS; i++) {
            const pointX = state.data[i][0] * SCALE + WIDTH / 2;
            const pointY = state.data[i][1] * SCALE + HEIGHT / 2;
            const distance = Math.sqrt((mousePos.x - pointX) ** 2 + (mousePos.y - pointY) ** 2);
            if (distance < 15 && distance < minDist) {
                minDist = distance;
                state.draggingIdx = i;
            }
        }
        if (state.draggingIdx !== -1) {
            canvas.style.cursor = "grabbing";
            drawAll();
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        if (state.draggingIdx === -1) return;
        const mousePos = getMousePos(canvas, e);
        state.data[state.draggingIdx][0] = (mousePos.x - WIDTH / 2) / SCALE;
        state.data[state.draggingIdx][1] = (mousePos.y - HEIGHT / 2) / SCALE;
        drawAll();
    });

    canvas.addEventListener("mouseup", () => {
        if (state.draggingIdx !== -1) {
            state.draggingIdx = -1;
            canvas.style.cursor = "default";
            drawAll();
        }
    });

    canvas.addEventListener("mouseleave", () => {
        if (state.draggingIdx !== -1) {
            state.draggingIdx = -1;
            canvas.style.cursor = "default";
            drawAll();
        }
    });

    document.getElementById("lcStep").addEventListener("click", singleStep);
    document.getElementById("lcStart").addEventListener("click", startTraining);
    document.getElementById("lcStop").addEventListener("click", stopTraining);
    document.getElementById("lcRandom").addEventListener("click", randomizeWeights);
    document.getElementById("lcRegC").addEventListener("input", function() {
        state.regC = parseFloat(this.value);
        drawAll();
    });
    document.getElementById("lcForm").addEventListener("change", function() {
        state.lossType = this.value;
        drawAll();
    });
    document.querySelectorAll("#weightMatrix input").forEach(input => {
        input.addEventListener("change", function() {
            updateParametersFromUI();
            drawAll();
        });
    });

    updateParameterInputs();
    drawAll();
}

if (document.readyState !== 'loading') {
    initLinearClassificationDemo();
} else {
    document.addEventListener('DOMContentLoaded', initLinearClassificationDemo);
}