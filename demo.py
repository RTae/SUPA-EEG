import os
import sys
import json
import base64
import io
import urllib.parse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import ThingsEEGDataset
from src.models.supaeeg import SUPAEEG
from src.encoders.vision_encoder import InternViTFeatureLookup
from src.utilities import Config, make_model

# ---------------------------------------------------------------------------
# HARDCODED CONFIGURATION
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "intra/supaeeg_intra_sub01.pt"  # Path to your trained model checkpoint
SUBJECT = 1                                       # Subject ID (1 to 10)
CONCEPT = "00197_wheelchair"                      # Target concept from the test set
# ---------------------------------------------------------------------------

def load_config() -> Config:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("conf/config.yaml")
    config = Config()
    for field_name in config.__dataclass_fields__:
        if hasattr(cfg, field_name):
            setattr(config, field_name, getattr(cfg, field_name))
    return config


def plot_eeg(eeg_tensor: torch.Tensor) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    data = eeg_tensor.numpy()
    for i in range(data.shape[0]):
        ax.plot(data[i] + i * 3.0, linewidth=1)
    ax.set_title("EEG Signal")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channels")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_inference(subject: int, target_concept: str, checkpoint_path: str) -> dict:
    config = load_config()
    
    # Load test dataset
    dataset = ThingsEEGDataset(
        dataset_dir=config.dataset_dir,
        data_type="test",
        subject=subject,
        load_images=False,
        data_average=config.data_average_test
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, device)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    # Find sample indices for the target concept
    indices = [i for i, c in enumerate(dataset.image_meta_data['test_img_concepts']) if c == target_concept]
    if not indices:
        raise ValueError(f"Concept '{target_concept}' not found in the test dataset split")
    
    # Load all EEG trials for this concept
    eeg_tensors = []
    target_file = None
    for idx in indices:
        eeg_tensor, _, _, _, _, _, img_file = dataset[idx]
        eeg_tensors.append(eeg_tensor)
        if target_file is None:
            target_file = img_file
            
    eeg_batch = torch.stack(eeg_tensors).to(device)  # (N_trials, 17, 100)
    
    # Compute average EEG embedding
    with torch.no_grad():
        zE_trials = model.embed(eeg_batch)  # (N_trials, 512)
        zE = torch.nn.functional.normalize(zE_trials.mean(dim=0, keepdim=True), dim=1).cpu().numpy()  # (1, 512)
        
    # Get test concept gallery
    concepts = sorted(list(set(dataset.image_meta_data['test_img_concepts'])))
    concept_to_file = {}
    for i in range(len(dataset.image_meta_data['test_img_concepts'])):
        c = dataset.image_meta_data['test_img_concepts'][i]
        f = dataset.image_meta_data['test_img_files'][i]
        if c not in concept_to_file:
            concept_to_file[c] = f
            
    # Retrieve & encode gallery image features
    feature_path = os.path.join(config.internvit_dir, "internvit_features.npy")
    lookup = InternViTFeatureLookup(feature_path=feature_path)
    files = [concept_to_file[c] for c in concepts]
    gallery_features = lookup.retrieve_batch(concepts, files)  # (200, 5, 3200)
    
    with torch.no_grad():
        zI = model.encode_image(gallery_features.to(device), subject_ids=None).cpu().numpy()  # (200, 512)
        
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(zE, zI)[0]
    
    top_indices = np.argsort(-sim)[:5]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "rank": rank,
            "concept": concepts[idx],
            "image_file": concept_to_file[concepts[idx]],
            "similarity": float(sim[idx])
        })
        
    return {
        "results": results,
        "target_file": target_file
    }

# ---------------------------------------------------------------------------
# HTTP Web Server
# ---------------------------------------------------------------------------

class DemoHTTPHandler(BaseHTTPRequestHandler):
        
    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)
        
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif path == "/api/meta":
            self.send_json({
                "subject": SUBJECT,
                "concept": CONCEPT,
                "checkpoint": CHECKPOINT_PATH
            })
            
        elif path == "/api/sample":
            try:
                config = load_config()
                dataset = ThingsEEGDataset(
                    dataset_dir=config.dataset_dir,
                    data_type="test",
                    subject=SUBJECT,
                    load_images=False,
                    data_average=config.data_average_test
                )
                
                # Find the correct concept index in the 200 test set concepts
                concept_idx = -1
                for i, c in enumerate(dataset.image_meta_data['test_img_concepts']):
                    if c == CONCEPT:
                        concept_idx = i
                        break
                if concept_idx == -1:
                    raise ValueError(f"Concept '{CONCEPT}' not found in the test dataset split")
                
                # Get the 80 trials for this concept (using the repetitions factor)
                indices = [concept_idx * dataset.number_of_repetitions + r for r in range(dataset.number_of_repetitions)]
                
                # Average the EEG traces across all trials for a clean ERP visualization
                eeg_traces = [dataset[idx][0] for idx in indices]
                eeg_average = torch.stack(eeg_traces).mean(dim=0)
                
                # Use the target image file from the first trial sample
                _, _, _, _, _, _, image_file = dataset[indices[0]]
                
                eeg_plot = plot_eeg(eeg_average)
                
                self.send_json({
                    "image_file": image_file,
                    "eeg_plot": eeg_plot
                })
            except Exception as e:
                self.send_error(500, str(e))
                
        elif path == "/api/decode":
            try:
                res = run_inference(SUBJECT, CONCEPT, CHECKPOINT_PATH)
                self.send_json(res)
            except Exception as e:
                self.send_json({"error": str(e)})
                
        elif path == "/api/image":
            concept = query.get("concept", [""])[0]
            image_file = query.get("file", [""])[0]
            
            project_root = os.path.dirname(os.path.abspath(__file__))
            img_dir = os.path.join(project_root, "data/things_eeg", "test_images")
            img_path = os.path.join(img_dir, concept, image_file)
            
            if os.path.isfile(img_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
            
    def send_json(self, data: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


# HTML UI Dashboard Template
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SUPAEEG Visual Decoding Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 30px;
            background-color: #f8f9fa;
            color: #333;
        }
        h1 {
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
        .meta-info {
            background-color: #e9ecef;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .column {
            flex: 1;
            background: #fff;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 12px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-bottom: 20px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .image-box {
            border: 1px solid #ccc;
            padding: 10px;
            text-align: center;
            background: #fafafa;
            margin-bottom: 20px;
        }
        .image-box img {
            max-width: 100%;
            max-height: 300px;
            display: block;
            margin: 10px auto;
        }
        .result-item {
            display: flex;
            align-items: center;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        .result-item img {
            width: 70px;
            height: 70px;
            object-fit: cover;
            margin-right: 15px;
            border: 1px solid #ccc;
        }
        .result-details {
            flex: 1;
        }
    </style>
</head>
<body>
    <h1>SUPAEEG Visual Decoding Demo</h1>
    
    <div class="meta-info" id="meta-info">
        Loading configuration...
    </div>
    
    <div class="container">
        <!-- Input Column -->
        <div class="column">
            <h2>Presented Stimulus & EEG</h2>
            <hr><br>
            
            <div class="image-box">
                <strong>EEG Signal Plot</strong>
                <div id="eeg-plot-container">Loading EEG...</div>
            </div>
            
            <div class="image-box">
                <strong>Target Image (Presented Stimulus)</strong>
                <div id="target-image-container">Loading Target Image...</div>
            </div>
        </div>
        
        <!-- Action & Output Column -->
        <div class="column">
            <h2>Decoder Inference</h2>
            <hr><br>
            
            <button id="btn-decode">Run Decoding Model</button>
            
            <h3>Top 5 Retrieved Images</h3>
            <div id="results-container">Click button to decode EEG signal.</div>
        </div>
    </div>

    <script>
        const metaInfo = document.getElementById('meta-info');
        const eegPlotContainer = document.getElementById('eeg-plot-container');
        const targetImageContainer = document.getElementById('target-image-container');
        const btnDecode = document.getElementById('btn-decode');
        const resultsContainer = document.getElementById('results-container');
        
        let targetConcept = "";
        
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const configRes = await fetch('/api/meta');
                const configData = await configRes.json();
                targetConcept = configData.concept;
                
                metaInfo.innerHTML = `
                    <strong>Subject:</strong> Subject ${configData.subject} | 
                    <strong>Target Concept:</strong> ${configData.concept} | 
                    <strong>Checkpoint:</strong> ${configData.checkpoint}
                `;
                
                const sampleRes = await fetch('/api/sample');
                const sampleData = await sampleRes.json();
                
                eegPlotContainer.innerHTML = `<img src="data:image/png;base64,${sampleData.eeg_plot}" alt="EEG">`;
                
                const imgUrl = `/api/image?concept=${configData.concept}&file=${sampleData.image_file}`;
                targetImageContainer.innerHTML = `<img src="${imgUrl}" alt="${configData.concept}"><br><strong>${configData.concept}</strong>`;
            } catch (e) {
                metaInfo.innerHTML = '<span style="color:red;">Failed to load server configuration. Make sure data paths are correct.</span>';
                eegPlotContainer.innerHTML = "Error loading EEG";
                targetImageContainer.innerHTML = "Error loading Image";
                console.error(e);
            }
        });
        
        btnDecode.addEventListener('click', async () => {
            btnDecode.disabled = true;
            btnDecode.innerText = "Decoding...";
            resultsContainer.innerHTML = "Running model inference...";
            
            try {
                const res = await fetch('/api/decode');
                const data = await res.json();
                
                if (data.error) {
                    resultsContainer.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
                    return;
                }
                
                resultsContainer.innerHTML = '';
                data.results.forEach(item => {
                    const imgUrl = `/api/image?concept=${item.concept}&file=${item.image_file}`;
                    const isCorrect = item.concept === targetConcept;
                    const style = isCorrect ? 'background-color: #d4edda; border: 1px solid #c3e6cb;' : '';
                    
                    const div = document.createElement('div');
                    div.className = 'result-item';
                    div.style = style;
                    div.innerHTML = `
                        <img src="${imgUrl}" alt="${item.concept}">
                        <div class="result-details">
                            <strong>Rank ${item.rank}: ${item.concept}</strong><br>
                            Similarity: ${item.similarity.toFixed(4)}
                        </div>
                    `;
                    resultsContainer.appendChild(div);
                });
            } catch (e) {
                resultsContainer.innerHTML = '<span style="color:red;">Error running decoding</span>';
                console.error(e);
            } finally {
                btnDecode.disabled = false;
                btnDecode.innerText = "Run Decoding Model";
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Runner setup
# ---------------------------------------------------------------------------

def main() -> None:
    port = 8080
    while port < 8100:
        try:
            server_address = ('', port)
            httpd = HTTPServer(server_address, DemoHTTPHandler)
            break
        except OSError:
            port += 1
            
    print(f"SUPAEEG visual decoding demo dashboard is ready at http://localhost:{port}/")
    print("Press Ctrl+C to terminate.")
    
    # Auto-open browser in background thread
    def open_browser():
        try:
            webbrowser.open(f"http://localhost:{port}/")
        except Exception:
            pass
    threading.Timer(1.0, open_browser).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
