let API_URL = 'http://10.8.0.1:30056';  // Changed to 'let' so it can be updated
let notificationContainer;

function updateApiUrl() {
    const newApiUrl = document.getElementById('apiUrl').value;
    if (newApiUrl) {
        API_URL = newApiUrl;
        showNotification(`API URL updated to: ${API_URL}`);
    } else {
        showNotification('Please enter a valid API URL.');
    }
}

function showHelpNotification(topic) {
    const helpMessages = {
        attributes: "Enter the attributes you want to use for classification, separated by commas.",
        priorities: "Enter the priorities for each attribute in the format 'attribute:priority', separated by commas. Higher numbers indicate higher priority.",
        apiUrl: "Enter the URL of the API server. This is used to generate the flowchart. Default is http://10.8.0.1:30056.",
        threshold: "Enter the priority threshold. Attributes with priority above this value will always be included and placed on top."
    };
    showNotification(helpMessages[topic]);
}

function createNotificationContainer() {
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.className = 'notification-container';
        document.body.appendChild(notificationContainer);
    }
}

function showNotification(message) {
    createNotificationContainer();

    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    notificationContainer.appendChild(notification);

    // Show the notification
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 100);

    // Hide and remove the notification after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            notificationContainer.removeChild(notification);
        }, 300); // Wait for fade-out transition to complete
    }, 5000);
}

function updateAttributes() {
    const attributes = document.getElementById('attributes').value.split(',').map(attr => attr.trim());
    if (attributes.length === 0 || (attributes.length === 1 && attributes[0] === '')) {
        showNotification("Please enter at least one attribute.");
        return;
    }

    const table = document.getElementById('dataTable');
    const headerRow = table.getElementsByTagName('thead')[0].getElementsByTagName('tr')[0];

    // Clear existing headers except for the checkbox and "Element"
    while (headerRow.cells.length > 2) {
        headerRow.deleteCell(2);
    }

    // Add new headers
    attributes.forEach(attr => {
        const th = document.createElement('th');
        th.textContent = attr;
        headerRow.appendChild(th);
    });

    // Update existing rows
    const tbody = table.getElementsByTagName('tbody')[0];
    for (let i = 0; i < tbody.rows.length; i++) {
        const row = tbody.rows[i];
        while (row.cells.length < attributes.length + 2) {
            const cell = row.insertCell(-1);
            cell.innerHTML = '<input type="text">';
        }
        while (row.cells.length > attributes.length + 2) {
            row.deleteCell(-1);
        }
    }
}

function addElement() {
    const table = document.getElementById('dataTable');
    const tbody = table.getElementsByTagName('tbody')[0];
    const row = tbody.insertRow();
    
    // Add checkbox cell
    const checkboxCell = row.insertCell(0);
    checkboxCell.innerHTML = '<input type="checkbox">';
    
    const elementNameCell = row.insertCell(1);
    elementNameCell.innerHTML = `<input type="text" value="Element ${tbody.rows.length}">`;
    
    for (let i = 2; i < table.rows[0].cells.length; i++) {
        const cell = row.insertCell();
        cell.innerHTML = '<input type="text">';
    }
}

function removeElement() {
    const table = document.getElementById('dataTable');
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = tbody.getElementsByTagName('tr');
    
    for (let i = rows.length - 1; i >= 0; i--) {
        const checkbox = rows[i].cells[0].getElementsByTagName('input')[0];
        if (checkbox.checked) {
            tbody.deleteRow(i);
        }
    }
}

async function generateFlowchart() {
    const attributes = document.getElementById('attributes').value;
    const priorities = document.getElementById('priorities').value;
    const threshold = parseFloat(document.getElementById('threshold').value);
    const data = getDataFromTable();
    const exportFormat = document.getElementById('exportFormat').value;
    const pngQuality = parseInt(document.getElementById('pngQuality').value);

    if (data.length === 0) {
        showNotification("Please enter some data in the table.");
        return;
    }

    try {
        // Step 1: Queue the request
        const queueResponse = await fetch(`${API_URL}/generate_flowchart`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                attributes,
                priorities,
                threshold,
                data,
                export_format: exportFormat,
                png_quality: pngQuality
            }),
        });

        if (!queueResponse.ok) {
            throw new Error(`HTTP error! status: ${queueResponse.status}`);
        }

        const queueResult = await queueResponse.json();
        const requestId = queueResult.request_id;

        // Step 2: Set up Server-Sent Events
        const eventSource = new EventSource(`${API_URL}/sse_result/${requestId}`);

        eventSource.addEventListener('result', (event) => {
            const result = JSON.parse(event.data);
            // Display the result
            document.getElementById('flowchart').src = `data:${result.content_type};base64,${result.flowchart}`;
            document.getElementById('flowchart').style.display = 'block';
            document.getElementById('outputText').value = `Accuracy: ${(result.accuracy * 100).toFixed(2)}%`;
            eventSource.close();
        });

        eventSource.addEventListener('processing', (event) => {
            const data = JSON.parse(event.data);
            document.getElementById('outputText').value = data.message;
        });

        eventSource.addEventListener('error', (event) => {
            const errorData = JSON.parse(event.data);
            console.error('Server error:', errorData);
            showNotification(`An error occurred: ${errorData}`);
            eventSource.close();
        });

        eventSource.onerror = (error) => {
            console.error('EventSource failed:', error);
            eventSource.close();
            showNotification('An error occurred while processing the request.');
        };

    } catch (error) {
        console.error('Error:', error);
        showNotification(`An error occurred: ${error.message}`);
    }
}


function togglePngQualityVisibility() {
    const exportFormat = document.getElementById('exportFormat').value;
    const pngQualityGroup = document.getElementById('pngQualityGroup');
    pngQualityGroup.style.display = exportFormat === 'png' ? 'block' : 'none';
}

function getDataFromTable() {
    const table = document.getElementById('dataTable');
    const data = [];
    const headers = Array.from(table.rows[0].cells).map(cell => cell.textContent).slice(2);

    for (let i = 1; i < table.rows.length; i++) {
        const row = table.rows[i];
        const rowData = {
            Element: row.cells[1].querySelector('input').value,
            attributes: {}
        };

        for (let j = 2; j < row.cells.length; j++) {
            const value = row.cells[j].querySelector('input').value.trim();
            if (value === '') {
                showNotification(`Row ${i} has missing information. Please fill all fields.`);
                return [];
            }
            rowData.attributes[headers[j-2]] = value;
        }

        data.push(rowData);
    }

    return data;
}

function saveConfiguration() {
    const config = {
        attributes: document.getElementById('attributes').value,
        priorities: document.getElementById('priorities').value,
        threshold: document.getElementById('threshold').value,
        data: getDataFromTable()
    };

    const jsonString = JSON.stringify(config, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const downloadLink = document.createElement('a');
    downloadLink.href = url;
    downloadLink.download = 'mineral_identification_config.json';
    
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);

    URL.revokeObjectURL(url);

    showNotification('Configuration saved successfully. Check your downloads folder.');
}

function loadConfiguration() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';

    fileInput.onchange = function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = function(e) {
            try {
                const config = JSON.parse(e.target.result);
                
                document.getElementById('attributes').value = config.attributes;
                document.getElementById('priorities').value = config.priorities;
                document.getElementById('threshold').value = config.threshold;
                
                updateAttributes();
                
                const table = document.getElementById('dataTable');
                const tbody = table.getElementsByTagName('tbody')[0];
                tbody.innerHTML = '';
                
                config.data.forEach(item => {
                    const row = tbody.insertRow();
                    
                    // Add checkbox cell
                    const checkboxCell = row.insertCell();
                    checkboxCell.innerHTML = '<input type="checkbox">';
                    
                    // Add element name cell
                    const elementCell = row.insertCell();
                    elementCell.innerHTML = `<input type="text" value="${item.Element}">`;
                    
                    // Add attribute cells
                    Object.values(item.attributes).forEach(value => {
                        const cell = row.insertCell();
                        cell.innerHTML = `<input type="text" value="${value}">`;
                    });
                });

                showNotification('Configuration loaded successfully');
            } catch (error) {
                console.error('Error:', error);
                showNotification(`An error occurred while loading the configuration: ${error.message}`);
            }
        };

        reader.readAsText(file);
    };

    fileInput.click();
}

function exportGraph() {
    const img = document.getElementById('flowchart');
    const exportFormat = document.getElementById('exportFormat').value;
    
    const link = document.createElement('a');
    link.download = `flowchart.${exportFormat}`;
    link.href = img.src;
    link.click();
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    updateAttributes();
    addElement();
    
    // Add event listener for export format change
    document.getElementById('exportFormat').addEventListener('change', togglePngQualityVisibility);
    togglePngQualityVisibility(); // Initial visibility setup
});