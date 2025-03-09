"""
FlowchartPlatform Backend
A FastAPI application for generating flowcharts based on decision trees.
This module provides a REST API that processes data, builds a decision tree,
and generates visual flowcharts for classification and decision-making.
"""
# ----- 1. IMPORTS -----
# Standard library imports - for core functionality
import asyncio                 # For asynchronous operations and event loops
import base64                  # For encoding binary data as base64 strings
import io                      # For in-memory file operations
import json                    # For JSON serialization and deserialization
import logging                 # For application logging
import multiprocessing         # For parallel processing using multiple CPU cores
import random                  # For random sampling and selection
import time                    # For time-related operations
import uuid                    # For generating unique identifiers
from collections import Counter # For counting occurrences in collections
from concurrent.futures import ThreadPoolExecutor  # For managing thread pools
import concurrent.futures      # For working with futures in concurrent code
from multiprocessing import Process  # For creating separate processes

# Third-party imports - external dependencies
import numpy as np             # For numerical operations
import pandas as pd            # For data manipulation and analysis
import uvicorn                 # ASGI server for running FastAPI applications
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks  # Web framework components
from fastapi.middleware.cors import CORSMiddleware  # For handling Cross-Origin Resource Sharing
from graphviz import Digraph   # For creating directed graphs/flowcharts
from pydantic import BaseModel, constr  # For data validation and settings management
from sse_starlette.sse import EventSourceResponse  # For Server-Sent Events
from typing import List, Dict, Optional  # For type annotations

# ----- 2. CONFIGURATION AND SETUP -----
# Set up logging - Configure logging with INFO level for application monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI application setup - Initialize the FastAPI application
app = FastAPI()

# CORS middleware configuration - Allow cross-origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (could be restricted in production)
    allow_credentials=True,       # Allow cookies in cross-origin requests
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# Worker configuration - Setup for parallel processing
NUM_WORKERS = 2  # Adjust based on your CPU cores and workload
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)  # For Windows compatibility
results = {}  # Shared dictionary for storing processing results

# ----- 3. DATA MODELS -----
class DecisionNode:
    """
    Represents a node in the decision tree.
    Each node either tests a feature or provides a value (leaf node).
    """
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature      # The attribute to test at this node
        self.children = children or {}  # Dictionary mapping feature values to child nodes
        self.value = value          # For leaf nodes, the classification value

class DataItem(BaseModel):
    """
    Model for individual data items.
    Each item has a main element and a dictionary of attributes.
    """
    Element: str  # The target element/class name
    attributes: Dict[str, Optional[str]]  # Map of attribute names to their values (can be None)

class GenerateFlowchartRequest(BaseModel):
    """
    Request model for flowchart generation.
    Contains all parameters needed to generate a flowchart.
    """
    attributes: constr(min_length=1)  # Comma-separated list of attributes to consider
    priorities: str                   # Priority weights for attributes in format "attr:weight,attr2:weight2"
    threshold: float                  # Threshold for using priority over information gain
    data: List[DataItem]              # List of data items to build the decision tree from
    export_format: str = "png"        # Output format (png or svg)
    png_quality: int = 300            # DPI for PNG export

# ----- 4. UTILITY FUNCTIONS -----
def parse_priorities(priorities_text):
    """
    Parse the priorities from text format to a dictionary.
    
    Args:
        priorities_text (str): Comma-separated string of attribute:priority pairs
        
    Returns:
        dict: Dictionary mapping attribute names to priority values
        
    Raises:
        ValueError: If the priority format is invalid
    """
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

# ----- 5. ALGORITHM FUNCTIONS -----
def entropy(labels):
    """
    Calculate entropy for a set of labels.
    Entropy measures the impurity or uncertainty in the dataset.
    
    Args:
        labels (array-like): List of class labels
        
    Returns:
        float: Entropy value (higher means more uncertainty)
    """
    counts = Counter(labels)  # Count occurrences of each unique label
    probabilities = [count / len(labels) for count in counts.values()]  # Calculate probability of each label
    return -sum(p * np.log2(p) for p in probabilities if p > 0)  # Shannon entropy formula

def information_gain(data, feature, target):
    """
    Calculate information gain for a specific feature.
    Information gain = entropy(parent) - weighted_average(entropy(children))
    
    Args:
        data (DataFrame): The dataset
        feature (str): The feature to calculate information gain for
        target (str): The target variable
        
    Returns:
        float: Information gain value (higher is better for decision splitting)
    """
    total_entropy = entropy(data[target])  # Entropy before split
    weighted_feature_entropy = 0
    
    # Calculate weighted entropy after splitting by each feature value
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)  # Proportion of samples with this feature value
        weighted_feature_entropy += weight * entropy(subset[target])
        
    # Information gain is the reduction in entropy
    return total_entropy - weighted_feature_entropy

def find_best_splits(data, features, target, feature_priorities):
    """
    Find the best features to split on based on information gain and priorities.
    
    Args:
        data (DataFrame): The dataset
        features (list): List of feature names to consider
        target (str): The target variable
        feature_priorities (dict): Dictionary mapping features to priority weights
        
    Returns:
        list: List of features with the highest (weighted) information gain
    """
    # Calculate information gain for each feature, weighted by its priority
    gains = {feature: information_gain(data, feature, target) * feature_priorities.get(feature, 1) 
            for feature in features if feature_priorities.get(feature, 1) != 0}
    
    if not gains:  # No valid features to split on
        return []
        
    max_gain = max(gains.values())  # Find highest gain value
    
    # Return all features that have the maximum gain (could be multiple)
    return [feature for feature, gain in gains.items() if gain == max_gain]

def build_tree(data, features, target, feature_priorities, priority_threshold, depth=0, max_depth=10):
    """
    Build a decision tree recursively.
    
    Args:
        data (DataFrame): The dataset
        features (list): List of feature names to consider
        target (str): The target variable
        feature_priorities (dict): Dictionary mapping features to priority weights
        priority_threshold (float): Threshold for using priority over information gain
        depth (int): Current depth in the tree
        max_depth (int): Maximum depth to build the tree to
        
    Returns:
        DecisionNode: Root node of the decision tree
    """
    # Base case 1: If all samples belong to one class, return a leaf node
    if len(data[target].unique()) == 1:
        return DecisionNode(value=data[target].iloc[0])
        
    # Base case 2: If we've run out of features or hit max depth, return a leaf node
    if len(features) == 0 or depth == max_depth:
        unique_values = data[target].unique()
        # If only one class, return it; otherwise return list of possible classes
        return DecisionNode(value=unique_values[0] if len(unique_values) == 1 else unique_values.tolist())

    # Find features with priority above threshold
    high_priority_features = [f for f in features if feature_priorities.get(f, 1) > priority_threshold]
    
    if high_priority_features:
        # If high priority features exist, use the one with highest priority
        best_feature = max(high_priority_features, key=lambda f: feature_priorities.get(f, 1))
    else:
        # Otherwise use information gain to find best feature
        best_features = find_best_splits(data, features, target, feature_priorities)
        if not best_features:
            # If no good splits found, return leaf with all possible values
            return DecisionNode(value=data[target].unique().tolist())
        # Randomly choose among equally good features (to avoid bias)
        best_feature = random.choice(best_features)

    # Create decision node with the selected feature
    node = DecisionNode(feature=best_feature)
    
    # Remove the chosen feature from consideration for child nodes
    remaining_features = [f for f in features if f != best_feature]

    # For each possible value of the selected feature, create a child node
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        if len(subset) > 0:
            node.children[value] = build_tree(
                subset, 
                remaining_features, 
                target, 
                feature_priorities, 
                priority_threshold, 
                depth + 1, 
                max_depth
            )

    return node

def prune_single_item_chains(node):
    """
    Remove single-item chains in the decision tree.
    This simplifies the tree by eliminating unnecessary decision nodes.
    
    Args:
        node (DecisionNode): The node to prune
        
    Returns:
        DecisionNode: The pruned node
    """
    # Base case: leaf node
    if node.value is not None:
        return node
        
    # If node has only one child, replace with the child
    if len(node.children) == 1:
        child = list(node.children.values())[0]
        return prune_single_item_chains(child)
        
    # Recursively prune all children
    for value, child in node.children.items():
        node.children[value] = prune_single_item_chains(child)
        
    return node

def classify(sample, tree):
    """
    Classify a sample using the decision tree.
    
    Args:
        sample (dict): Dictionary of feature values
        tree (DecisionNode): The decision tree to use
        
    Returns:
        str or list: Predicted class or list of possible classes
    """
    # Base case: leaf node
    if tree.value is not None:
        return tree.value
        
    # Get the value for the feature at this node
    feature_value = sample.get(tree.feature)
    
    # If feature value not found in children, return None (can't classify)
    if feature_value not in tree.children:
        return None
        
    # Recursively classify using the appropriate child node
    return classify(sample, tree.children[feature_value])

def generate_orthogonal_flow_chart(tree, feature_priorities, priority_threshold, dot=None, parent_name=None, edge_label=None):
    """
    Generate a flowchart visualization from the decision tree.
    
    Args:
        tree (DecisionNode): The decision tree to visualize
        feature_priorities (dict): Dictionary mapping features to priority weights
        priority_threshold (float): Threshold for high-priority features
        dot (Digraph, optional): Existing Digraph object to add to
        parent_name (str, optional): Name of parent node (for edge connections)
        edge_label (str, optional): Label for edge from parent
        
    Returns:
        Digraph: The Graphviz Digraph object representing the flowchart
    """
    # Initialize a new Digraph if not provided
    if dot is None:
        dot = Digraph(comment='Mineral Identification Flow Chart')
        # Set graph attributes for visual style
        dot.attr(rankdir='TB', size='20,20', fontname='Arial', splines='ortho', nodesep='0.5', ranksep='0.5')

    # Create a unique identifier for this node
    node_name = str(id(tree))

    # Handle leaf nodes (with values)
    if tree.value is not None:
        if isinstance(tree.value, list):
            # For multiple possible values, create a central node with children
            dot.node(node_name, 'Possible Elements', shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
            for val in tree.value:
                child_node_name = f"{node_name}_{val}"
                dot.node(child_node_name, val, shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
                dot.edge(node_name, child_node_name)
        else:
            # For single value, create one leaf node
            dot.node(node_name, tree.value, shape='ellipse', style='filled', fillcolor='lightblue', fontname='Arial Bold')
    else:
        # For decision nodes, use different colors based on priority
        fillcolor = 'yellow' if feature_priorities.get(tree.feature, 0) > priority_threshold else 'lightgreen'
        dot.node(node_name, tree.feature, shape='rectangle', style='filled', fillcolor=fillcolor, fontname='Arial')

    # If this is a child node, connect to parent through intermediate decision diamond
    if parent_name:
        intermediate_node = f"{parent_name}_{node_name}"
        dot.node(intermediate_node, edge_label, shape='diamond', style='filled', fillcolor='lightgray', fontname='Arial', fontsize='10')
        dot.edge(parent_name, intermediate_node, arrowhead='none')
        dot.edge(intermediate_node, node_name)

    # Recursively add all children to the graph
    if tree.children:
        for value, child in tree.children.items():
            generate_orthogonal_flow_chart(child, feature_priorities, priority_threshold, dot, node_name, str(value))

    return dot

# ----- 6. REQUEST PROCESSING -----
def process_request(request):
    """
    Process a flowchart generation request.
    This is the main function that orchestrates the entire flowchart generation process.
    
    Args:
        request (GenerateFlowchartRequest): The request object with all parameters
        
    Returns:
        dict: Dictionary with generated flowchart, content type and accuracy
        
    Raises:
        Exception: Any error during processing
    """
    try:
        logger.info(f"Processing request: {request}")
        
        # Parse and prepare input data
        attributes = [attr.strip() for attr in request.attributes.split(',') if attr.strip()]
        attribute_priorities = parse_priorities(request.priorities)
        priority_threshold = request.threshold

        # Assign default priority 1.0 to any attribute without explicit priority
        for attr in attributes:
            if attr not in attribute_priorities:
                attribute_priorities[attr] = 1.0

        # Convert request data to format suitable for pandas DataFrame
        data = []
        for item in request.data:
            element = item.Element
            attrs = {attr: item.attributes.get(attr, 'unknown') for attr in attributes}
            data.append({"Element": element, **attrs})
        
        # Create DataFrame and handle missing values
        df = pd.DataFrame(data)
        df.fillna('unknown', inplace=True)

        # Build and optimize the decision tree
        tree = build_tree(df, attributes, "Element", attribute_priorities, priority_threshold)
        tree = prune_single_item_chains(tree)  # Simplify the tree

        # Generate the flowchart
        flow_chart = generate_orthogonal_flow_chart(tree, attribute_priorities, priority_threshold)
        
        # Export in the requested format (PNG or SVG)
        if request.export_format.lower() == "svg":
            img = flow_chart.pipe(format='svg')
            content_type = "image/svg+xml"
        else:
            flow_chart.attr(dpi=str(request.png_quality))  # Set resolution for PNG
            img = flow_chart.pipe(format='png')
            content_type = "image/png"

        # Encode the image as base64 for transmission
        encoded_img = base64.b64encode(img).decode('utf-8')

        # Calculate accuracy by testing the tree against original data
        total_score = 0
        for _, sample in df.iterrows():
            classification = classify(sample, tree)
            if classification is None:  # Skip if can't be classified
                continue
            if isinstance(classification, list):
                # Partial credit for list of possibilities containing correct answer
                if sample["Element"] in classification:
                    total_score += 1 / len(classification)
            else:
                # Full credit for exact match
                if classification == sample["Element"]:
                    total_score += 1

        # Calculate final accuracy percentage
        accuracy = total_score / len(df) if len(df) > 0 else 0

        logger.info(f"Request processed successfully. Accuracy: {accuracy}")
        return {
            "flowchart": encoded_img,
            "content_type": content_type,
            "accuracy": accuracy
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise  # Re-raise to be caught by worker process

# ----- 7. WORKER PROCESS MANAGEMENT -----
def worker_process(worker_id, request_queue, results):
    """
    Worker process function for handling requests.
    Runs in a separate process and processes requests from the queue.
    
    Args:
        worker_id (int): Identifier for this worker
        request_queue (Queue): Queue of requests to process
        results (dict): Shared dictionary for storing results
    """
    logger.info(f"Worker {worker_id} started")
    while True:
        # Get next request from queue (blocking)
        request_id, request = request_queue.get()
        
        # None request signals shutdown
        if request is None:
            logger.info(f"Worker {worker_id} shutting down")
            break
            
        try:
            # Process the request and store result
            result = process_request(request)
            results[request_id] = result
            logger.info(f"Worker {worker_id} completed request {request_id}")
        except Exception as e:
            # Store error if processing fails
            logger.error(f"Worker {worker_id} encountered an error: {str(e)}")
            results[request_id] = {"error": str(e)}

def start_worker_processes(request_queue, results):
    """
    Start the worker processes.
    
    Args:
        request_queue (Queue): Queue for requests
        results (dict): Shared dictionary for results
        
    Returns:
        list: List of started Process objects
    """
    worker_processes = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(i, request_queue, results))
        p.start()
        worker_processes.append(p)
    return worker_processes

def shutdown_workers(worker_processes, request_queue):
    """
    Shut down the worker processes gracefully.
    
    Args:
        worker_processes (list): List of Process objects
        request_queue (Queue): Queue for requests
    """
    # Send shutdown signal to all workers
    for _ in worker_processes:
        request_queue.put((None, None))
        
    # Wait for all workers to finish
    for p in worker_processes:
        p.join()

# ----- 8. API ENDPOINTS -----
@app.post("/generate_flowchart")
async def generate_flowchart(request: GenerateFlowchartRequest):
    """
    API endpoint for generating a flowchart.
    Queues the request for processing by a worker.
    
    Args:
        request (GenerateFlowchartRequest): The flowchart generation request
        
    Returns:
        dict: Dictionary with request ID and status message
        
    Raises:
        HTTPException: If there's an error queuing the request
    """
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Submit request to thread pool executor
        future = executor.submit(process_request, request)
        results[request_id] = future
        
        logger.info(f"Request {request_id} queued for processing")
        return {"request_id": request_id, "message": "Request queued for processing"}
    except Exception as e:
        logger.error(f"Error queueing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sse_result/{request_id}")
async def sse_result(request: Request, request_id: str):
    """
    Server-Sent Events endpoint for retrieving results.
    Streams processing status and final results to the client.
    
    Args:
        request (Request): The HTTP request object
        request_id (str): ID of the request to check
        
    Returns:
        EventSourceResponse: SSE stream with processing updates
    """
    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            if request_id in results:
                future = results[request_id]
                
                # If the result is a Future (still processing)
                if isinstance(future, concurrent.futures.Future):
                    if future.done():
                        try:
                            # Get result and remove from results dict to free memory
                            result = future.result()
                            del results[request_id]
                            yield {
                                "event": "result",
                                "data": json.dumps(result)
                            }
                            break
                        except Exception as e:
                            # If error occurred during processing
                            yield {
                                "event": "error",
                                "data": str(e)
                            }
                            break
                else:
                    # If the result is already available (not a future)
                    yield {
                        "event": "result",
                        "data": json.dumps(future)
                    }
                    del results[request_id]
                    break
            else:
                # If request is not found, it's still being processed
                yield {
                    "event": "processing",
                    "data": json.dumps({"message": "Processing..."})
                }
            
            # Wait before checking again
            await asyncio.sleep(1)

    # Return SSE response with the event generator
    return EventSourceResponse(event_generator())

# ----- 9. APPLICATION STARTUP -----
def run():
    """
    Run the FastAPI application.
    Sets up multiprocessing and starts the server.
    """
    # Create multiprocessing manager for shared data structures
    manager = multiprocessing.Manager()
    request_queue = manager.Queue()  # Queue for requests
    results_shared = manager.dict()  # Dictionary for results
    
    # Store in application state for access in other parts of the app
    app.state.request_queue = request_queue
    app.state.results = results_shared
    
    # Start worker processes
    worker_processes = start_worker_processes(request_queue, results_shared)
    
    try:
        # Start ASGI server
        uvicorn.run(app, host="0.0.0.0", port=8080)
    finally:
        # Ensure workers are shut down properly
        shutdown_workers(worker_processes, request_queue)

if __name__ == "__main__":
    run()
