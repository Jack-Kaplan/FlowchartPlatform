import threading
import requests
import time

API_URL = 'http://localhost:8000'  # Update this to your server's URL if different

def simulate_user(user_id):
    print(f"User {user_id} starting")

    # Sample data payload matching your website's expected input
    attributes = "Color,Hardness,Luster"
    priorities = "Color:1,Hardness:2,Luster:3"
    threshold = 1
    data = [
        {
            "Element": "Quartz",
            "attributes": {
                "Color": "Colorless",
                "Hardness": "7",
                "Luster": "Vitreous"
            }
        },
        {
            "Element": "Calcite",
            "attributes": {
                "Color": "White",
                "Hardness": "3",
                "Luster": "Vitreous"
            }
        },
        {
            "Element": "Gypsum",
            "attributes": {
                "Color": "White",
                "Hardness": "2",
                "Luster": "Silky"
            }
        },
        # Add more sample minerals if needed
    ]
    export_format = "png"
    png_quality = 300

    payload = {
        'attributes': attributes,
        'priorities': priorities,
        'threshold': threshold,
        'data': data,
        'export_format': export_format,
        'png_quality': png_quality
    }

    try:
        # Step 1: Send POST request to queue the flowchart generation
        response = requests.post(f"{API_URL}/generate_flowchart", json=payload)
        if response.status_code != 200:
            print(f"User {user_id}: Error in POST request: {response.status_code}")
            return
        result = response.json()
        request_id = result.get('request_id')
        if not request_id:
            print(f"User {user_id}: No request_id in response")
            return

        # Step 2: Poll for the result until processing is complete
        while True:
            time.sleep(1)  # Wait 1 second before polling again
            result_response = requests.get(f"{API_URL}/get_result/{request_id}")
            if result_response.status_code != 200:
                print(f"User {user_id}: Error in GET request: {result_response.status_code}")
                continue
            result = result_response.json()
            if result.get('message') != "Processing not complete":
                # Processing is complete
                print(f"User {user_id}: Processing complete")
                break
    except Exception as e:
        print(f"User {user_id}: Exception occurred: {e}")

    print(f"User {user_id} completed")

def main():
    import sys
    if len(sys.argv) < 2:
        n = int(input("Enter number of users to simulate: "))
    else:
        n = int(sys.argv[1])

    threads = []
    for i in range(n):
        t = threading.Thread(target=simulate_user, args=(i+1,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("All users have completed")

if __name__ == '__main__':
    main()
