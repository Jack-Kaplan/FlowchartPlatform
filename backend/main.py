from multiprocessing import Process
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from typing import List, Dict, Optional
import pandas as pd
import random
from collections import Counter
import numpy as np
from graphviz import Digraph
import base64
import io
import multiprocessing
import time
import uuid
import logging
import uvicorn
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
import asyncio
import json
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Number of worker processes
NUM_WORKERS = 2  # Adjust based on your CPU

# Use ThreadPoolExecutor instead of ProcessPoolExecutor for Windows compatibility
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

# Shared dictionary for results (thread-safe)
results = {}

class DecisionNode:
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature
        self.children = children or {}
        self.value = value

class DataItem(BaseModel):
    Element: str
    attributes: Dict[str, Optional[str]]  # Allow attribute values to be None

class GenerateFlowchartRequest(BaseModel):
    attributes: constr(min_length=1)  # Ensures attributes is a non-empty string
    priorities: str
    threshold: float
    data: List[DataItem]
    export_format: str = "png"
    png_quality: int = 300

def entropy(labels):
    counts = Counter(labels)
    probabilities = [count / len(labels) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_feature_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)
        weighted_feature_entropy += weight * entropy(subset[target])
    return total_entropy - weighted_feature_entropy

def find_best_splits(data, features, target, feature_priorities):
    gains = {feature: information_gain(data, feature, target) * feature_priorities.get(feature, 1) for feature in features if feature_priorities.get(feature, 1) != 0}
    if not gains:
        return []
    max_gain = max(gains.values())
    return [feature for feature, gain in gains.items() if gain == max_gain]

def prune_single_item_chains(node):
    if node.value is not None:
        return node
    if len(node.children) == 1:
        child = list(node.children.values())[0]
        return prune_single_item_chains(child)
    for value, child in node.children.items():
        node.children[value] = prune_single_item_chains(child)
    return node

def build_tree(data, features, target, feature_priorities, priority_threshold, depth=0, max_depth=10):
    if len(data[target].unique()) == 1:
        return DecisionNode(value=data[target].iloc[0])
    if len(features) == 0 or depth == max_depth:
        unique_values = data[target].unique()
        return DecisionNode(value=unique_values[0] if len(unique_values) == 1 else unique_values.tolist())

    high_priority_features = [f for f in features if feature_priorities.get(f, 1) > priority_threshold]
    if high_priority_features:
        best_feature = max(high_priority_features, key=lambda f: feature_priorities.get(f, 1))
    else:
        best_features = find_best_splits(data, features, target, feature_priorities)
        if not best_features:
            return DecisionNode(value=data[target].unique().tolist())
        best_feature = random.choice(best_features)

    node = DecisionNode(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        if len(subset) > 0:
            node.children[value] = build_tree(subset, remaining_features, target, feature_priorities, priority_threshold, depth + 1, max_depth)

    return node

def classify(sample, tree):
    if tree.value is not None:
        return tree.value
    feature_value = sample.get(tree.feature)
    if feature_value not in tree.children:
        return None
    return classify(sample, tree.children[feature_value])

def generate_orthogonal_flow_chart(tree, feature_priorities, priority_threshold, dot=None, parent_name=None, edge_label=None):
    if dot is None:
        dot = Digraph(comment='Mineral Identification Flow Chart')
        dot.attr(rankdir='TB', size='20,20', fontname='Arial', splines='ortho', nodesep='0.5', ranksep='0.5')

    node_name = str(id(tree))

    if tree.value is not None:
        if isinstance(tree.value, list):
            dot.node(node_name, 'Possible Elements', shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
            for val in tree.value:
                child_node_name = f"{node_name}_{val}"
                dot.node(child_node_name, val, shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
                dot.edge(node_name, child_node_name)
        else:
            dot.node(node_name, tree.value, shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
    else:
        fillcolor = 'yellow' if feature_priorities.get(tree.feature, 0) > priority_threshold else 'lightgreen'
        dot.node(node_name, tree.feature, shape='rectangle', style='filled', fillcolor=fillcolor, fontname='Arial')

    if parent_name:
        intermediate_node = f"{parent_name}_{node_name}"
        dot.node(intermediate_node, edge_label, shape='diamond', style='filled', fillcolor='lightgray', fontname='Arial', fontsize='10')
        dot.edge(parent_name, intermediate_node, arrowhead='none')
        dot.edge(intermediate_node, node_name)

    if tree.children:
        for value, child in tree.children.items():
            generate_orthogonal_flow_chart(child, feature_priorities, priority_threshold, dot, node_name, str(value))

    return dot

def parse_priorities(priorities_text):
    attribute_priorities = {}
    if priorities_text:
        for item in priorities_text.split(','):
            try:
                attribute, priority = item.split(':')
                priority = float(priority.strip())
                if priority != 0:
                    attribute_priorities[attribute.strip()] = priority
            except ValueError:
                raise ValueError(f"Invalid priority format: {item}")
    return attribute_priorities

def process_request(request):
    try:
        logger.info(f"Processing request: {request}")
        attributes = [attr.strip() for attr in request.attributes.split(',') if attr.strip()]
        # Removed manual validation
        attribute_priorities = parse_priorities(request.priorities)
        priority_threshold = request.threshold

        for attr in attributes:
            if attr not in attribute_priorities:
                attribute_priorities[attr] = 1.0

        data = []
        for item in request.data:
            element = item.Element
            attrs = {attr: item.attributes.get(attr, 'unknown') for attr in attributes}
            data.append({"Element": element, **attrs})
        
        df = pd.DataFrame(data)

        df.fillna('unknown', inplace=True)

        tree = build_tree(df, attributes, "Element", attribute_priorities, priority_threshold)
        tree = prune_single_item_chains(tree)

        flow_chart = generate_orthogonal_flow_chart(tree, attribute_priorities, priority_threshold)
        
        if request.export_format.lower() == "svg":
            img = flow_chart.pipe(format='svg')
            content_type = "image/svg+xml"
        else:
            flow_chart.attr(dpi=str(request.png_quality))
            img = flow_chart.pipe(format='png')
            content_type = "image/png"

        encoded_img = base64.b64encode(img).decode('utf-8')

        total_score = 0
        for _, sample in df.iterrows():
            classification = classify(sample, tree)
            if classification is None:
                continue
            if isinstance(classification, list):
                if sample["Element"] in classification:
                    total_score += 1 / len(classification)
            else:
                if classification == sample["Element"]:
                    total_score += 1

        accuracy = total_score / len(df) if len(df) > 0 else 0

        logger.info(f"Request processed successfully. Accuracy: {accuracy}")
        return {
            "flowchart": encoded_img,
            "content_type": content_type,
            "accuracy": accuracy
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise

def worker_process(worker_id, request_queue, results):
    logger.info(f"Worker {worker_id} started")
    while True:
        request_id, request = request_queue.get()
        if request is None:
            logger.info(f"Worker {worker_id} shutting down")
            break
        try:
            result = process_request(request)
            results[request_id] = result
            logger.info(f"Worker {worker_id} completed request {request_id}")
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered an error: {str(e)}")
            results[request_id] = {"error": str(e)}

@app.post("/generate_flowchart")
async def generate_flowchart(request: GenerateFlowchartRequest):
    try:
        request_id = str(uuid.uuid4())
        future = executor.submit(process_request, request)
        results[request_id] = future
        logger.info(f"Request {request_id} queued for processing")
        return {"request_id": request_id, "message": "Request queued for processing"}
    except Exception as e:
        logger.error(f"Error queueing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sse_result/{request_id}")
async def sse_result(request: Request, request_id: str):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            if request_id in results:
                future = results[request_id]
                if isinstance(future, concurrent.futures.Future):
                    if future.done():
                        try:
                            result = future.result()
                            del results[request_id]
                            yield {
                                "event": "result",
                                "data": json.dumps(result)
                            }
                            break
                        except Exception as e:
                            yield {
                                "event": "error",
                                "data": str(e)
                            }
                            break
                else:
                    # If the result is already available
                    yield {
                        "event": "result",
                        "data": json.dumps(future)
                    }
                    del results[request_id]
                    break
            else:
                yield {
                    "event": "processing",
                    "data": json.dumps({"message": "Processing..."})
                }
            
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

def start_worker_processes(request_queue, results):
    worker_processes = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, request_queue, results))
        p.start()
        worker_processes.append(p)
    return worker_processes

def shutdown_workers(worker_processes, request_queue):
    for _ in worker_processes:
        request_queue.put((None, None))
    for p in worker_processes:
        p.join()

def run():
    manager = multiprocessing.Manager()
    request_queue = manager.Queue()
    results_shared = manager.dict()
    
    app.state.request_queue = request_queue
    app.state.results = results_shared
    
    worker_processes = start_worker_processes(request_queue, results_shared)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        shutdown_workers(worker_processes, request_queue)

if __name__ == "__main__":
    run()
