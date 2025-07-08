function initKNNDemo() {
    console.log("Initializing KNN Demo...");
    
    // Setup container
    const container = document.getElementById('simulationContainer');
    container.innerHTML = `
        <div class="knn-container" style="max-width: 900px; margin: 0 auto;">
            <h3 class="text-center mb-3 main_color">K-Nearest Neighbors Interactive Demo</h3>
            
            <div class="row">
                <div class="col-lg-8 order-lg-1 order-2">
                    <canvas id="knnCanvas" width="600" height="450" 
                            style="border:1px solid #e6e6e6; background:white; max-width: 100%;"></canvas>
                    <p class="text-muted small mt-2 secondary_font">Drag points to see classification changes</p>
                </div>
                <div class="col-lg-4 order-lg-2 order-1 mb-3">
                    <div class="controls card p-3" style="background-color: #f8f9fa; border: 1px solid #e6e6e6;">
                        <div class="form-group text-center">
                            <label for="metricSelect" class="secondary_font"><strong>Distance Metric:</strong></label>
                            <select id="metricSelect" class="form-control">
                                <option value="euclidean">L2 (Euclidean)</option>
                                <option value="manhattan">L1 (Manhattan)</option>
                            </select>
                        </div>
                        <div class="form-group text-center">
                            <label for="kSlider" class="secondary_font"><strong>Num Neighbors (K):</strong> 
                                <span id="kValue" class="badge" style="background-color: #8C1515;">5</span>
                            </label>
                            <input type="range" id="kSlider" min="1" max="15" value="5" class="form-control">
                        </div>
                        <div class="form-group text-center">
                            <label for="classSlider" class="secondary_font"><strong>Num Classes:</strong> 
                                <span id="classValue" class="badge" style="background-color: #8C1515;">3</span>
                            </label>
                            <input type="range" id="classSlider" min="2" max="5" value="3" class="form-control">
                        </div>
                        <div class="form-group text-center">
                            <label for="pointSlider" class="secondary_font"><strong>Num Points:</strong> 
                                <span id="pointValue" class="badge" style="background-color: #8C1515;">60</span>
                            </label>
                            <input type="range" id="pointSlider" min="10" max="200" value="60" class="form-control">
                        </div>
                        <button id="resetBtn" class="btn btn-block mt-2" 
                                style="background-color: #8C1515; color: white;">Regenerate Points</button>
                    </div>
                </div>
            </div>
            
            <div class="description mt-4 p-3 rounded" style="background-color: #f8f9fa; border: 1px solid #e6e6e6;">
                <p class="secondary_color">This interactive demo explores the K-Nearest Neighbors algorithm for classification.</p>
                <p class="secondary_color">Each point in the plane is colored with the class that would be assigned to it using the K-Nearest Neighbors algorithm. Points for which the K-Nearest Neighbor algorithm results in a tie are colored white.</p>
                <p class="mb-0"><small><em>Inspired by the <a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/" target="_blank" style="color: #8C1515;">EECS 498: Deep Learning for Computer Vision</a> course at the University of Michigan.</em></small></p>
            </div>
        </div>
    `;

    // Canvas setup
    const canvas = document.getElementById('knnCanvas');
    const ctx = canvas.getContext('2d');

    // Demo state
    const state = {
        k: 5,
        numClasses: 3,
        numPoints: 60,
        metric: 'euclidean',
        points: [],
        colors: ['#FF5252', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0']
    };

    // Distance metrics
    const metrics = {
        euclidean: (a, b) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2),
        manhattan: (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1])
    };

    // Generate random points in clusters
    function generatePoints() {
        state.points = [];
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(canvas.width, canvas.height) * 0.4;
        
        // Create clusters
        for (let c = 0; c < state.numClasses; c++) {
            const angle = (c * (2 * Math.PI / state.numClasses)) - Math.PI/2;
            const clusterX = centerX + Math.cos(angle) * radius * 0.6;
            const clusterY = centerY + Math.sin(angle) * radius * 0.6;
            
            const pointsPerClass = Math.floor(state.numPoints / state.numClasses);
            for (let i = 0; i < pointsPerClass; i++) {
                state.points.push([
                    clusterX + (Math.random() - 0.5) * radius * 0.5,
                    clusterY + (Math.random() - 0.5) * radius * 0.5,
                    c
                ]);
            }
        }
        
        // Add remaining points randomly
        while (state.points.length < state.numPoints) {
            state.points.push([
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.floor(Math.random() * state.numClasses)
            ]);
        }
    }

    // Find k-nearest neighbors
    function getNeighbors(point) {
        return _.chain(state.points)
            .map(p => ({ point: p, dist: metrics[state.metric](point, p) }))
            .sortBy('dist')
            .first(state.k)
            .pluck('point')
            .value();
    }

    // Draw everything
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw decision boundaries
        const step = 10;
        for (let x = 0; x < canvas.width; x += step) {
            for (let y = 0; y < canvas.height; y += step) {
                const neighbors = getNeighbors([x, y]);
                const votes = _.countBy(neighbors, p => p[2]);
                const maxClass = _.max(Object.keys(votes), k => votes[k]);
                
                // Check for ties
                const isTie = Object.keys(votes).filter(k => votes[k] === votes[maxClass]).length > 1;
                
                ctx.fillStyle = isTie ? '#FFFFFF80' : state.colors[maxClass] + '80';
                ctx.fillRect(x - step/2, y - step/2, step, step);
            }
        }
        
        // Draw points
        state.points.forEach(p => {
            ctx.fillStyle = state.colors[p[2]];
            ctx.beginPath();
            ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    // Event listeners for controls
    document.getElementById('metricSelect').addEventListener('change', function() {
        state.metric = this.value.split(' ')[0];
        draw();
    });

    document.getElementById('kSlider').addEventListener('input', function() {
        state.k = parseInt(this.value);
        document.getElementById('kValue').textContent = state.k;
        draw();
    });

    document.getElementById('classSlider').addEventListener('input', function() {
        state.numClasses = parseInt(this.value);
        document.getElementById('classValue').textContent = state.numClasses;
        generatePoints();
        draw();
    });

    document.getElementById('pointSlider').addEventListener('input', function() {
        state.numPoints = parseInt(this.value);
        document.getElementById('pointValue').textContent = state.numPoints;
        generatePoints();
        draw();
    });

    document.getElementById('resetBtn').addEventListener('click', function() {
        generatePoints();
        draw();
    });

    // Handle point dragging
    let draggedPoint = null;
    let isDragging = false;

    function getMousePos(canvas, evt) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (evt.clientX - rect.left) * (canvas.width / rect.width),
            y: (evt.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    canvas.addEventListener('mousedown', function(e) {
        const mousePos = getMousePos(canvas, e);
        const clickRadius = 10 * (canvas.width / 600); // Scale with canvas size
        
        // Find all points within the click radius
        const nearbyPoints = state.points.filter(p => 
            metrics[state.metric]([mousePos.x, mousePos.y], p) < clickRadius
        );
        
        if (nearbyPoints.length > 0) {
            // Select the closest point among nearby points
            state.draggedPoint = _.min(nearbyPoints, p => 
                metrics[state.metric]([mousePos.x, mousePos.y], p)
            );
            state.isDragging = true;
            canvas.style.cursor = 'grabbing';
        }
    });

    canvas.addEventListener('mousemove', function(e) {
        if (!state.isDragging || !state.draggedPoint) return;
        
        const mousePos = getMousePos(canvas, e);
        
        // Constrain to canvas bounds
        state.draggedPoint[0] = Math.max(0, Math.min(canvas.width, mousePos.x));
        state.draggedPoint[1] = Math.max(0, Math.min(canvas.height, mousePos.y));
        
        draw();
    });

    canvas.addEventListener('mouseup', function() {
        state.isDragging = false;
        state.draggedPoint = null;
        canvas.style.cursor = 'default';
    });

    canvas.addEventListener('mouseleave', function() {
        state.isDragging = false;
        state.draggedPoint = null;
        canvas.style.cursor = 'default';
    });

    // Initialize
    generatePoints();
    draw();
    console.log("KNN Demo Ready!");
}

// Start the demo when loaded
if (document.readyState !== 'loading') {
    initKNNDemo();
} else {
    document.addEventListener('DOMContentLoaded', initKNNDemo);
}