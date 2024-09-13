from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import random
from collections import Counter
import numpy as np
from graphviz import Digraph
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class DecisionNode:
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature
        self.children = children or {}
        self.value = value

class DataItem(BaseModel):
    Element: str
    attributes: Dict[str, str]

class Configuration(BaseModel):
    attributes: str
    priorities: str
    data: List[DataItem]

class GenerateFlowchartRequest(BaseModel):
    attributes: str
    priorities: str
    threshold: float
    data: List[DataItem]
    export_format: str = "png"  # Can be "png" or "svg"
    png_quality: int = 300  # DPI for PNG export, ignored for SVG

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
    gains = {feature: information_gain(data, feature, target) * feature_priorities.get(feature, 1) for feature in features}
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
        return DecisionNode(value=data[target].mode().iloc[0])

    high_priority_features = [f for f in features if feature_priorities.get(f, 0) > priority_threshold]
    if high_priority_features:
        best_feature = max(high_priority_features, key=lambda f: feature_priorities.get(f, 0))
    else:
        best_features = find_best_splits(data, features, target, feature_priorities)
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
    feature_value = sample[tree.feature]
    if feature_value not in tree.children:
        return None  # Unable to classify
    return classify(sample, tree.children[feature_value])

def generate_orthogonal_flow_chart(tree, feature_priorities, priority_threshold, dot=None, parent_name=None, edge_label=None):
    if dot is None:
        dot = Digraph(comment='Mineral Identification Flow Chart')
        dot.attr(rankdir='TB', size='20,20', fontname='Arial', splines='ortho', nodesep='0.5', ranksep='0.5')

    node_name = str(id(tree))

    if tree.value is not None:
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
        high_priority_children = [child for child in tree.children.values() if child.feature and feature_priorities.get(child.feature, 0) > priority_threshold]
        normal_children = [child for child in tree.children.values() if child not in high_priority_children]

        if high_priority_children:
            with dot.subgraph() as s:
                s.attr(rank='same')
                for child in high_priority_children:
                    child_name = str(id(child))
                    s.node(child_name)

        if normal_children:
            with dot.subgraph() as s:
                s.attr(rank='same')
                for child in normal_children:
                    child_name = str(id(child))
                    s.node(child_name)

        for value, child in tree.children.items():
            generate_orthogonal_flow_chart(child, feature_priorities, priority_threshold, dot, node_name, str(value))

    return dot

def parse_priorities(priorities_text):
    attribute_priorities = {}
    if priorities_text:
        for item in priorities_text.split(','):
            try:
                attribute, priority = item.split(':')
                attribute_priorities[attribute.strip()] = float(priority.strip())
            except ValueError:
                raise ValueError(f"Invalid priority format: {item}")
    return attribute_priorities

@app.post("/generate_flowchart")
async def generate_flowchart(request: GenerateFlowchartRequest):
    try:
        attributes = [attr.strip() for attr in request.attributes.split(',') if attr.strip()]
        if not attributes:
            raise HTTPException(status_code=400, detail="Please enter at least one attribute.")

        attribute_priorities = parse_priorities(request.priorities)
        priority_threshold = request.threshold

        for attr in attributes:
            if attr not in attribute_priorities:
                attribute_priorities[attr] = 1.0

        data = [{"Element": item.Element, **item.attributes} for item in request.data]
        df = pd.DataFrame(data)

        tree = build_tree(df, attributes, "Element", attribute_priorities, priority_threshold)
        tree = prune_single_item_chains(tree)

        flow_chart = generate_orthogonal_flow_chart(tree, attribute_priorities, priority_threshold)
        
        if request.export_format.lower() == "svg":
            img = flow_chart.pipe(format='svg')
            content_type = "image/svg+xml"
        else:  # Default to PNG
            flow_chart.attr(dpi=str(request.png_quality))
            img = flow_chart.pipe(format='png')
            content_type = "image/png"

        encoded_img = base64.b64encode(img).decode('utf-8')

        correct = sum(1 for _, sample in df.iterrows() if classify(sample, tree) == sample["Element"])
        accuracy = correct / len(df)

        return {
            "flowchart": encoded_img,
            "content_type": content_type,
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_configuration")
async def save_configuration(config: Configuration):
    try:
        return {"message": "Configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_configuration")
async def load_configuration(config: Configuration):
    try:
        return {"message": "Configuration loaded successfully", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)