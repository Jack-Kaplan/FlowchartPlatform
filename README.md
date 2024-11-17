# FlowchartPlatform

FlowchartPlatform is a FastAPI-based tool for generating orthogonal flowcharts from structured datasets. It is designed to classify elements based on user-defined attributes and priorities, leveraging decision tree logic. The tool produces high-quality visualizations in PNG or SVG formats and supports asynchronous processing for scalable performance.

---

## Features

- **Decision Tree Logic**: Builds a decision tree from structured data using entropy and information gain.
- **Customizable Priorities**: Assign weights to attributes to prioritize them in decision-making.
- **Flowchart Visualization**: Generates orthogonal flowcharts in PNG or SVG formats.
- **Accuracy Calculation**: Evaluates classification accuracy based on input data.
- **Asynchronous Processing**: Handles requests asynchronously for scalability.
- **SSE Support**: Provides real-time updates via Server-Sent Events (SSE).
- **Thread-safe Shared Memory**: Uses shared dictionaries for storing results across worker processes.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jack-Kaplan/FlowchartPlatform.git
   cd FlowchartPlatform
   ```

2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Graphviz Separately**  
   FlowchartPlatform uses Graphviz for generating flowcharts. Make sure Graphviz is installed on your system. You can install it using your package manager or download it from [Graphviz.org](https://graphviz.org/download/).

   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install graphviz
     ```
   - **macOS** (with Homebrew):
     ```bash
     brew install graphviz
     ```
   - **Windows**: Download and install Graphviz from [Graphviz.org](https://graphviz.org/download/).

4. Start the server:
   ```bash
   python main.py
   ```

---

## API Endpoints

### 1. **Generate Flowchart**

**Endpoint**: `/generate_flowchart`  
**Method**: POST  
**Description**: Submits a request to generate a flowchart from structured data.

**Request Body**:
```json
{
  "attributes": "attribute1,attribute2,...",
  "priorities": "attribute1:priority1,attribute2:priority2,...",
  "threshold": 0.5,
  "data": [
    {
      "Element": "ElementName",
      "attributes": {
        "attribute1": "value1",
        "attribute2": "value2"
      }
    },
    ...
  ],
  "export_format": "png",
  "png_quality": 300
}
```

**Response**:
```json
{
  "request_id": "unique-request-id",
  "message": "Request queued for processing"
}
```

### 2. **SSE Result**

**Endpoint**: `/sse_result/{request_id}`  
**Method**: GET  
**Description**: Streams updates about the processing status and final result.

**Response**:
- `event: result` with the flowchart data and accuracy.
- `event: error` if an error occurs during processing.
- `event: processing` while the request is being processed.

---

## Usage Example

1. Send a request to `/generate_flowchart` with your data and configuration.
2. Use the `request_id` from the response to fetch updates via `/sse_result/{request_id}`.
3. The final result will include the flowchart (as base64-encoded data) and the classification accuracy.

---

## Environment Configuration

- **NUM_WORKERS**: Adjust the number of worker processes in the `main.py` file based on your system's CPU.

---

## Licensing

- **Personal (Non-Commercial) Use**: This project is free for personal use.  
- **Commercial Use**: For any commercial use, please reach out for permission.

---

## Development

1. Fork and clone the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Submit a pull request with your changes.

---

Feel free to contribute or open issues at [FlowchartPlatform](https://github.com/Jack-Kaplan/FlowchartPlatform).
